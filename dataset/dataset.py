import json
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, BatchSampler, RandomSampler, SequentialSampler, Sampler
import os
from transformers import AutoTokenizer
from model.hub import get_backbone_id


class CustomDistributedBatchSampler(DistributedSampler): # 完全可用
    def __init__(self, dataset, num_replicas = None, rank = None, shuffle = True, seed = 0, drop_last = False, max_batch_length=3000, max_length=800, num_buckets=10):
        # DistributedSampler queries dist.get_world_size(), which fails when no
        # process group has been initialized (single process / CPU smoke runs).
        if num_replicas is None or rank is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
                rank = dist.get_rank() if rank is None else rank
            else:
                num_replicas = 1 if num_replicas is None else num_replicas
                rank = 0 if rank is None else rank
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last) # Initialize the DistributedSampler

        self.max_batch_length = max_batch_length
        self.max_length = max_length
        self.num_buckets = num_buckets

        self.bucket = [([], 0) for _ in range(self.num_buckets)]

    def __iter__(self):
        iterator = super().__iter__()

        for idx in iterator:
            
            length = self.dataset.get_data_len(idx)

            if length is None:
                continue

            if length > self.max_length:
                print('长度过长')
                continue

            bucket_id = min(int(length / (self.max_length / self.num_buckets)), self.num_buckets - 1)

            # 取出旧桶
            idxs, total_len = self.bucket[bucket_id]

            # 如果加一条就爆，先 yield 旧桶
            if total_len + length > self.max_batch_length and idxs:
                yield idxs

                self.bucket[bucket_id] = ([], 0)
                idxs, total_len = [], 0
                # self.bucket[bucket_id] = (idxs + [idx], total_len + length)
            # 把当前样本塞进桶
            self.bucket[bucket_id] = (idxs + [idx], total_len + length)

        # epoch 结束：把剩余桶全 yield（除非 drop_last）
        if not self.drop_last:
            for idxs, _ in self.bucket:
                if idxs:

                    yield idxs
        # 如果 drop_last=True，这里就不 yield 了
                
    def __len__(self):
        # return 1000000
        raise TypeError(f"{self.__class__.__name__} does not have a defined length.")
    
class CustomBatchSampler(Sampler): # 单卡
    def __init__(self, dataset, num_replicas = None, rank = None, shuffle = True, seed = 0, drop_last = False, max_batch_length=3000, max_length=1000, num_buckets=10):
        # super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last) # Initialize the DistributedSampler
        super().__init__(dataset)
        self.dataset = dataset
        self.max_batch_length = max_batch_length
        self.max_length = max_length
        self.num_buckets = num_buckets
        self.drop_last = drop_last
        self.bucket = [([], 0) for _ in range(self.num_buckets)]

    def __iter__(self):
        # import pdb;pdb.set_trace()
        n = len(self.dataset)
        indices = list(range(n))
        for idx in indices:
            length = self.dataset.get_data_len(idx)
            if length is None:
                continue
            if length > self.max_length:
                continue

            bucket_id = min(int(length / (self.max_length / self.num_buckets)), self.num_buckets - 1)

            # 取出旧桶
            idxs, total_len = self.bucket[bucket_id]

            # 如果加一条就爆，先 yield 旧桶
            if total_len + length > self.max_batch_length and idxs:

                # print(f'桶 cost_Time: {cost_time}s')
                yield idxs
                self.bucket[bucket_id] = ([], 0)
                idxs, total_len = [], 0

            # 把当前样本塞进桶
            self.bucket[bucket_id] = (idxs + [idx], total_len + length)

        # epoch 结束：把剩余桶全 yield（除非 drop_last）
        if not self.drop_last:
            for idxs, _ in self.bucket:
                if idxs:
                    yield idxs
        # 如果 drop_last=True，这里就不 yield 了
                
    def __len__(self):
        raise TypeError(f"{self.__class__.__name__} does not have a defined length.")      


# ===================== 2. 改写后的JSONL Dataset【✅核心优化：apply_chat_template】 =====================
class ChatJsonlDataset(Dataset):
    def __init__(self, jsonl_file_path, tokenizer, shuffle_on_load=False, seed=42, max_seq_len=2000):
        super().__init__()
        assert os.path.exists(jsonl_file_path), f"文件不存在: {jsonl_file_path}"
        self.jsonl_file = jsonl_file_path
        self.max_seq_len = max_seq_len
        self.data_list = self._load_jsonl_data()
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id

        self.shuffle_on_load = shuffle_on_load  # 是否加载时打乱
        self.seed = seed
        
        # 加载数据并可选打乱
        self.data_list = self._load_jsonl_data()
        if self.shuffle_on_load:
            self._shuffle_data()
            
    def _load_jsonl_data(self):
        """加载jsonl数据，每行是一个包含messages的dict"""
        data_list = []
        with open(self.jsonl_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    json_obj = json.loads(line)
                    if "messages" in json_obj and isinstance(json_obj["messages"], list):
                        data_list.append(json_obj["messages"])
                except json.JSONDecodeError as e:
                    print(f"跳过第{line_idx+1}行，解析失败: {e}")
        assert len(data_list) > 0, "JSONL文件中无有效数据！"
        return data_list
    
    def _shuffle_data(self):
        """打乱data_list（仅单卡可用）"""
        import random
        random.seed(self.seed)
        random.shuffle(self.data_list)
        print(f"Dataset数据已打乱（种子：{self.seed}）")

    def shuffle(self, seed=None):
        """手动调用打乱数据（用于多epoch）"""
        if seed is not None:
            self.seed = seed
        self._shuffle_data()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """✅ 核心重构：使用 apply_chat_template 完成标准格式拼接+tokenize+标签掩码，完美适配Qwen3"""
        messages = self.data_list[idx]
        # import pdb;pdb.set_trace()
        user_text = '<|im_start|>user\n' + messages[0]['content'] + '<|im_end|>\n'
        assistant_text = '<|im_start|>assistant\n' + messages[1]['content'] + '<|im_end|>\n' 

        user_token = self.tokenizer(user_text, return_tensors='pt', add_special_tokens=True).input_ids.squeeze(0) # (T1)
        assistant_token = self.tokenizer(assistant_text, return_tensors='pt', add_special_tokens=True).input_ids.squeeze(0) # (T2)
        # import pdb;pdb.set_trace()

        return {
            "user_tokens": user_token,
            "assistant_tokens": assistant_token,
        }
    
    def get_data_len(self, idx):
        messages = self.data_list[idx]
        user_text = '<|im_start|>user\n' + messages[0]['content'] + '<|im_end|>\n'
        user_token = self.tokenizer(user_text, return_tensors='pt', add_special_tokens=True).input_ids.squeeze()

        return user_token.size(0)
    # ===================== 3. Collate_fn 【✅仅1处小修改：PAD_TOKEN_ID改为自动获取】 =====================
    def chat_custom_collate_fn(self, batch):
        user_tokens_list = [sample["user_tokens"] for sample in batch]
        assistant_tokens_list = [sample["assistant_tokens"] for sample in batch]

        batch_seq_user = [len(ids) for ids in user_tokens_list]
        batch_seq_assistant = [len(ids) for ids in assistant_tokens_list]

        # print(batch_seq_assistant)
        batch_max_len_user = max(batch_seq_user)
        batch_max_len_assistant = max(batch_seq_assistant)
        batch_input_ids = []
        batch_target_ids = []
        batch_attention_mask = []


        for i in range(len(batch)):
            user_tokens = user_tokens_list[i]
            assistant_tokens = assistant_tokens_list[i]
            pad_len_user = batch_max_len_user - len(user_tokens)
            pad_len_assistant = batch_max_len_assistant - len(assistant_tokens)

            padded_input_ids = torch.cat([user_tokens, torch.tensor([self.pad_token_id]*pad_len_user, dtype=torch.long)])
            padded_target_ids = torch.cat([assistant_tokens, torch.tensor([self.pad_token_id]*pad_len_assistant, dtype=torch.long)])
            attention_mask = torch.cat([torch.ones_like(assistant_tokens), torch.zeros(pad_len_assistant, dtype=torch.long)])

            batch_input_ids.append(padded_input_ids)
            batch_target_ids.append(padded_target_ids)
            batch_attention_mask.append(attention_mask)
        # import pdb;pdb.set_trace()
        return {
            "tokens": torch.stack(batch_input_ids, dim=0),
            "target_ids": torch.stack(batch_target_ids, dim=0),
            "attention_mask": torch.stack(batch_attention_mask, dim=0),
        }

# ===================== 4. 分布式采样器 DataLoader 封装（无任何修改，完全保留你的代码） =====================
def build_chat_dataloader(
    jsonl_path,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    is_distributed=True
):
    tokenizer = AutoTokenizer.from_pretrained(
    get_backbone_id(),
    padding_side="right",    # 右侧补零，LLM训练标准配置
    truncation_side="right", # 右侧截断
)
    dataset = ChatJsonlDataset(jsonl_file_path=jsonl_path, tokenizer=tokenizer, max_seq_len=2000)
    MAX_SEQ_LENGTH = 1500
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    
    # batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
    batch_sampler = CustomBatchSampler(
        dataset=dataset,
        shuffle=shuffle,
        drop_last=drop_last,
        max_batch_length=3000,
        max_length=MAX_SEQ_LENGTH,
        num_buckets=10
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=dataset.chat_custom_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=lambda _: torch.manual_seed(42)
    )
    return dataloader, sampler

# ===================== 5. 调试测试代码 【✅保留所有你的逻辑+断点+路径，无修改】 =====================
if __name__ == "__main__":
    TEST_JSONL_PATH = "train_data/example_data"
    BATCH_SIZE = 2                         
    IS_DISTRIBUTED = False                 
    SHUFFLE_DATA = False                   
    NUM_WORKERS = 0       


    print("="*60)
    print("开始构建数据集和DataLoader...")
    dataloader, sampler = build_chat_dataloader(
        jsonl_path=TEST_JSONL_PATH,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_DATA,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        is_distributed=IS_DISTRIBUTED
    )
    print(f"✅ 构建完成！数据集总样本数: {len(dataloader.dataset)}")
    print("="*60)
    for batch_idx, batch_data in enumerate(dataloader):
        # print(f"\n===== 当前遍历到第 {batch_idx+1} 个Batch =====")
        
        tokens = batch_data["tokens"]
        target_ids = batch_data["target_ids"]
        attention_mask = batch_data["attention_mask"]

        
        print(f"input_ids  shape: {tokens.shape}      -> [batch_size, seq_len]")
        # print(f"attention_mask shape: {attention_mask.shape} -> [batch_size, seq_len]")

        
        # print("\n▸ input_ids 前5个token内容:\n", tokens[:, :5])
        # print("\n▸ attention_mask 前5位:\n", attention_mask[:, :5])

        
        # print("\n▸ 第1条样本的文本还原(前10个token):")
        # first_sample_ids = tokens[0].cpu().numpy().tolist()
        # first_sample_ids = [x for x in first_sample_ids if x != PAD_TOKEN_ID]
        # decode_text = tokenizer.decode(first_sample_ids, skip_special_tokens=False)
        # print(f"还原文本: {decode_text[:200]}")
        
        # if batch_idx >= 1:
        #     print("\n⚠️  调试模式：只遍历前2个Batch，如需遍历全部请注释该行！")
        #     break
    
    print("\n" + "="*60)
    print("✅ DataLoader遍历调试完成！")
    print("="*60)