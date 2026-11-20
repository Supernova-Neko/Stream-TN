from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

from model.hub import get_model_id


class stream_tn_model(torch.nn.Module):
    def __init__(self, model_id: str | None = None):
        super().__init__()
        resolved = get_model_id(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(resolved)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        # self.lm_head = self.model.lm_head
        self.eos_token_id = self.tokenizer.eos_token_id

    def forward(self, 
                tokens, 
                target_ids,
                attention_mask=None,
                ):
        # import pdb;pdb.set_trace()
        
        delay_frame = 4 
        dtype, device = next(self.parameters()).dtype, tokens.device
        b, s = tokens.shape 
        # input_ids = tokens[:, :-1]
        tokens_embeds = self.model.get_input_embeddings()(tokens)
        target_ids_embeds = self.model.get_input_embeddings()(target_ids)

        pad_1frame = (0, 0, delay_frame, 0, 0, 0)
        target_ids_embeds_padded = torch.nn.functional.pad(target_ids_embeds, pad_1frame, mode="constant", value=0.0) # [B, T2+delay_frame, D]
        max_seq_len = max(tokens_embeds.shape[1], target_ids_embeds_padded.shape[1])
        tokens_embeds_final = torch.nn.functional.pad(
            tokens_embeds, 
            (0,0, 0, max_seq_len - tokens_embeds.shape[1], 0,0), 
            mode="constant", value=0.0
        ) # [B, max_seq_len, D]
        target_ids_embeds_final = torch.nn.functional.pad(
            target_ids_embeds_padded, 
            (0,0, 0, max_seq_len - target_ids_embeds_padded.shape[1], 0,0), 
            mode="constant", value=0.0
        ) # [B, max_seq_len, D]
        pad_target_ids = torch.nn.functional.pad(target_ids, pad=(delay_frame, 0), mode="constant", value=0)
        target_ids_final = torch.nn.functional.pad(
            pad_target_ids,
            (0, max_seq_len - pad_target_ids.shape[1]),
            mode="constant", value=0
        ) # [B, max_seq_len]
        attention_mask_padded = torch.nn.functional.pad(attention_mask, pad=(delay_frame, 0), mode="constant", value=1.0)
        pad_zero_len = max_seq_len - attention_mask_padded.shape[1]
        attention_mask_final = torch.nn.functional.pad(attention_mask_padded, pad=(0, pad_zero_len), mode="constant", value=0.0)  # [B, max_seq_len]

        total_emb = tokens_embeds_final + target_ids_embeds_final # [B, max_seq_len, D]
        input_emb = total_emb[:, :-1, :] # [B, max_seq_len-1, D]
        attention_mask_in = attention_mask_final[:, :-1] if attention_mask_final is not None else None
        targets = target_ids_final[:, delay_frame:]
        attention_mask_target = attention_mask_final[:, delay_frame:] if attention_mask_final is not None else None

        # import pdb;pdb.set_trace()
        input_pos = torch.arange(0, max_seq_len - 1, dtype=torch.long, device=input_emb.device).unsqueeze(0).expand(b, max_seq_len - 1)
        seq_len = attention_mask_in.sum(dim=-1) # 长度为S - 1
        curr_backbone_mask = torch.arange(0, seq_len.max(), device=seq_len.device)[None, :].expand(b, seq_len.max())
        curr_backbone_mask = curr_backbone_mask < seq_len[:, None] # [b, 1, s]
        h = self.model(inputs_embeds=input_emb, attention_mask=curr_backbone_mask, position_ids=input_pos)


        logits = h.logits.to(dtype=dtype)
        logits = logits[:, delay_frame - 1:, :]
        # loss = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")

        active_targets = torch.where(attention_mask_target == 1, targets, torch.tensor(-100, device=device, dtype=targets.dtype))


        # 步骤2：展平维度 + 用ignore_index=-100忽略padding位 → 只算有效位loss
        # logits展平：[b*(s-1), vocab_size]，target展平：[b*(s-1)]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  # 必须展平，cross_entropy的shape要求
            active_targets.reshape(-1),
            ignore_index=-100,  # 核心参数：忽略所有标签=-100的位置
            reduction="mean"    # loss为有效位的平均损失，合理！
        )


        # 1. 从logits取出预测值：取最后一维的最大值下标 → 预测的token_id，shape: [b, s-1]
        preds = torch.argmax(logits, dim=-1)
        # 2. 计算【预测正确的位置】：预测值 == 真实值，返回布尔矩阵 [b, s-1]
        correct_preds = (preds == targets)
        # 3. 只保留有效位的预测结果：用attention_mask_target过滤padding位，padding位不计入统计
        if attention_mask_target is not None:
            correct_preds = correct_preds & attention_mask_target.bool()
            # 有效token总数
            total_valid_tokens = attention_mask_target.sum()
        else:
            # 无掩码时，所有token都算有效
            total_valid_tokens = torch.numel(targets)
        # 4. 计算准确率：正确数 / 有效总数 (防止除零错误，加个极小值)
        total_correct_tokens = correct_preds.sum()
        token_acc = total_correct_tokens / torch.clamp(total_valid_tokens, min=1)
        # loss_fct = transformers.CrossEntropyLoss()
        # loss = loss_fct(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1))
        outputs = {'loss': loss, 'accuracy': token_acc}

        return outputs
    
    
    def generate(self, 
                token: torch.Tensor,
                ) -> torch.Tensor:
        
        # import pdb;pdb.set_trace()
        
        delay_frame = 4
        self.model.eval()
        cache = DynamicCache(config=self.model.config)
        
        token_embeds = self.model.get_input_embeddings()(token) # (b, t, d)
        # 1. 定义要pad的帧数（10帧）
        pad_frames = 10

        # 2. 创建正确维度的空白帧（shape [b, 10, d]）
        pad_embeds = torch.zeros(
            token_embeds.size(0),  # b: 批次维度，和原张量一致
            pad_frames,            # 10: 要pad的时间帧数
            token_embeds.size(2),  # d: 嵌入维度，和原张量一致
            device=token_embeds.device,
            dtype=token_embeds.dtype
        )

        token_embeds_padded = torch.cat([token_embeds, pad_embeds], dim=1) # (b, t+10, d)

        
        input_token = token[:, :delay_frame] # (b, 1)
        start_generate_time = time.time()
        inputs_embeds = self.model.get_input_embeddings()(input_token) # (b, 1, d)

        # inputs_embeds = token_embeds_padded[:, :delay_frame, :]  # (b, delay_frame, d)
        curr_backbone_mask = None
        input_pos = torch.arange(delay_frame).unsqueeze(0).long().to(token.device) # (1, delay_frame)
        result_token = []
        # import pdb; pdb.set_trace()
        first_flag = False
        for i in range(delay_frame, token_embeds_padded.shape[1]): # (1, n)
            
            hidden = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=curr_backbone_mask,
                position_ids=input_pos,
                use_cache=True,
                past_key_values=cache,
            )
            logits = hidden.logits[:,-1:] # (b, 1, vocab_size)
            next_token = torch.argmax(logits, dim=-1).squeeze(0)  # (b,)
            if first_flag == False:
                first_flag = True
                end_generate_time = time.time()
                cost_time = end_generate_time - start_generate_time
                # print(f"✅ First token generated: {next_token.item()}, time taken: {time.time() - start_generate_time:.4f} seconds")
            if next_token == self.eos_token_id:
                break
            result_token.append(next_token)

            next_token_embed = self.model.get_input_embeddings()(next_token.unsqueeze(0)) # (b, 1, d)
            inputs_embeds = next_token_embed + token_embeds_padded[:, i:i + 1, :]  # (b, 1, d)
            input_pos = input_pos[:,-1:] + 1


        return result_token, cost_time