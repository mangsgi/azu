import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import get_peft_model, LoraConfig, PeftModel, prepare_model_for_kbit_training

import gc
import csv
import shutil
import base64
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from dotenv import load_dotenv

# 조각화 완화
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# .env 파일에서 Hugging Face 토큰 로드 (Qwen 모델 접근에 필요)
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("Warning: Hugging Face 토큰이 .env 파일에 설정되지 않았습니다. 비공개 모델 로드 시 오류가 발생할 수 있습니다.")

# GPU 환경 확인
if torch.cuda.is_available():
    print(f"현재 사용중인 GPU   : {torch.cuda.get_device_name(0)}")
    print(f"현재 torch 버전    : {torch.version.cuda}")
    print(f"현재 cuDNN 버전    : {torch.backends.cudnn.version()}")
else:
    print("GPU를 사용할 수 없습니다. CPU로 실행됩니다.")

# Image 픽셀 제약 풀기
Image.MAX_IMAGE_PIXELS = None

# --- 1. 설정 (Configuration) ---
class Config:
    # 데이터 경로
    TRAIN_DATA_PATH = 'data/deep_chal_multitask_dataset.parquet'
    TEST_DATA_PATH = 'data/deep_chal_multitask_dataset_test.parquet'
    
    # 디렉토리 경로
    BASE_OUTPUT_DIR = './results_qwen' # 출력 디렉토리 변경
    SUBMISSION_FILE = 'submission.csv'

    # 모델 변경
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Partial Training을 위한 파라미터
    TASKS = ['captioning', 'vqa', 'math_reasoning', 'summarization', 'text_qa']
    SUBSAMPLE_FRAC = 0.005   # 각 task별 10%만 학습
    SEED = 42              # 재현 가능성 확보용 시드

    # PEFT (LoRA) 설정
    LORA_R = 16 # VRAM 사용량을 고려하여 R값 조정 (64 -> 16)
    LORA_ALPHA = 32 # R의 두 배로 설정하는 것이 일반적
    LORA_DROPOUT = 0.05
    
    # Qwen 모델에 맞는 타겟 모듈로 변경
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]

    # 학습 하이퍼파라미터
    NUM_TRAIN_EPOCHS = 1 
    PER_DEVICE_TRAIN_BATCH_SIZE = 1 # 모델이 커졌으므로 배치 사이즈 조정 (16 -> 4)
    GRADIENT_ACCUMULATION_STEPS = 4 # 실질 배치 사이즈 = 2 * 8 = 16
    LEARNING_RATE = 2e-5 # 일반적인 LoRA 학습률로 조정
    OPTIM = "paged_adamw_8bit" # "adamw_8bit" 
    LR_SCHEDULER_TYPE = "cosine"
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    LOGGING_STEPS = 50
    
    # 추론 관련
    MAX_NEW_TOKENS = 1024 

    # 태스크 목록
    TASKS = ['captioning', 'vqa', 'math_reasoning', 'summarization', 'text_qa']

    # 병합된 어댑터 저장 경로
    MERGED_ADAPTER_PATH = os.path.join(BASE_OUTPUT_DIR, "merged_adapter")

# --- 2. 데이터셋 및 전처리 ---
def get_task_instruction(sample: dict) -> str:
    task = (sample.get('task') or '').strip()
    inp = sample.get('input', '') or ''
    q = sample.get('question', '') or ''

    EXAMPLE_TEXT_QA_BRIEF = (
        "Example (short):\n"
        "Context: Eva left Seoul at 08:30, reached Busan at 12:10, and met Jin at Central Station.\n"
        "Questions: ['Who left Seoul?', 'When did she leave?', 'Where did she arrive?', 'When did she arrive?', 'Whom did she meet?', 'Where did they meet?']\n"
        "Expected JSON: {\"input_text\": [\"Eva\",\"08:30\",\"Busan\",\"12:10\",\"Jin\",\"Central Station\"], "
        "\"answer_start\": [0,18,33,42,57,64], \"answer_end\": [3,23,38,47,60,79]}\n"
    )
    
    if task == 'captioning':
        return (
            "Describe the image in natural English as a short paragraph (3–5 sentences). "
            "Avoid headings or bullets. If clearly readable text appears, quote it exactly."
        )
    elif task == 'vqa':
        return (
            "Answer the question about the image in a few words. "
            "Use a number for counts, 'yes' or 'no' for yes/no. "
            "If the answer is not visible or unclear, return 'unknown'.\n"
            f"Question: {q}"
        )
    elif task == 'math_reasoning':
        return (
            "Solve the problem briefly.\n\n"
            "For every calculation and expression, put the arithmetic inside double angle bracket. e.g., <<24*2=48>>"
            "At the end, output only the final numeric answer after four hash marks, e.g., #### 16 (no extra text).\n\n"
            f"Problem:\n{inp}"
        )
    elif task == 'summarization':
        return (
            "Summarize the text in 3–5 sentences in neutral English. Keep key entities, actions, dates, and numbers."
            f"\n\nText:\n{inp}"
        )
    elif task == 'text_qa':
        return (
            "Answer each question using ONLY the given context."
            "Extract the SHORTEST exact substring(s) from the context for each question and return a JSON with "
            '"input_text" (answers in order), "answer_start" (0-based), "answer_end" (exclusive).\n\n'
            + EXAMPLE_TEXT_QA_BRIEF +
            f"Context:\n{inp}\n\nQuestions:\n{q}"
        )
    else:
        # 안전망 (FINAL_SUFFIX 참조 제거)
        return f"Process the input and provide the best possible answer.\n{inp or q}"

class MultitaskDataset(Dataset):
    def __init__(self, df, processor, is_train=True):
        self.df = df
        self.processor = processor
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_input):
        try:
            if isinstance(image_input, str) and image_input.startswith("http"):
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(image_input, timeout=15, headers=headers)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            elif isinstance(image_input, bytes):
                return Image.open(BytesIO(image_input)).convert("RGB")
            elif isinstance(image_input, str): # base64 encoded
                return Image.open(BytesIO(base64.b64decode(image_input))).convert("RGB")
        except Exception:
            return None # 이미지 로드 실패 시 None 반환
        return None

    def __getitem__(self, idx):
        sample = self.df.iloc[idx].to_dict()
        instruction = get_task_instruction(sample)
        output = sample.get('output', '')
        
        image = None
        if sample.get('input_type') == 'image':
            image = self._load_image(sample.get('input'))
        
        # Qwen Chat Template에 맞게 메시지 구성
        if image:
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ]
        else:
            user_content = [{"type": "text", "text": instruction}]

        if self.is_train:
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": output}]}
            ]
        else:
            messages = [{"role": "user", "content": user_content}]
        
        item = {
            "messages": messages, 
            "image": image, 
            "input_type": sample.get('input_type') # 이 줄을 추가하여 input_type을 넘겨줍니다.
        }
        if not self.is_train:
            item["id"] = sample.get("ID")
        return item

# --- 3. 개별 태스크 학습 함수 ---
def train_task_adapter(config: Config, task: str, full_train_df: pd.DataFrame):
    print("\n" + "="*80)
    print(f"### STARTING TRAINING FOR TASK: [ {task.upper()} ] ###")
    print("="*80)

    task_df = full_train_df[full_train_df['task'] == task].reset_index(drop=True)
    if len(task_df) == 0:
        print(f"No data for task: {task}. Skipping.")
        return None

    # 서브샘플링만 수행 (프리필터 제거)
    orig_len = len(task_df)
    frac = getattr(config, "SUBSAMPLE_FRAC", 0.1)
    seed = getattr(config, "SEED", 42)
    if 0 < frac < 1:
        n = max(1, int(round(orig_len * frac)))
        task_df = task_df.sample(n=n, random_state=seed).reset_index(drop=True)
        print(f"Subsampled '{task}': {orig_len} -> {len(task_df)} rows (frac={frac})")
    else:
        print(f"Subsampling disabled; using all {orig_len} rows.")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,    
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    processor = AutoProcessor.from_pretrained(config.MODEL_ID, token=hf_token, trust_remote_code=True)

    # 데이터 전처리 패딩
    def vl_collate_fn(features):
        """ 이미지 로딩에 실패한 데이터를 배치에서 제외합니다. """
        original_size = len(features)
        
        # 텍스트 태스크이거나(input_type != 'image'), 이미지가 성공적으로 로드된 경우에만 데이터를 유지합니다.
        features = [
            f for f in features 
            if f['input_type'] != 'image' or f.get('image') is not None
        ]
        
        # 만약 배치에서 데이터가 제외되었다면 경고 메시지를 출력할 수 있습니다.
        if len(features) < original_size:
            print(f"Warning: Dropped {original_size - len(features)} samples from batch due to image loading errors.")

        # 핵심: 배치가 비면 더미 배치를 반환 → loss에 기여 0 (no-op)
        if not features:
            import torch
            pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
            input_ids = torch.tensor([[pad_id]], dtype=torch.long)
            attention_mask = torch.tensor([[0]], dtype=torch.long)   # 마스킹 0 → 모델이 보지 않음
            labels = torch.full_like(input_ids, -100)                # -100 → loss 계산에서 완전 제외
            print("[collate] Empty batch -> injected dummy no-op batch")
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


        # 1) 메시지를 텍스트로 변환 (훈련이니 generation 프롬프트는 False)
        texts = [processor.apply_chat_template(
            f["messages"], tokenize=False, add_generation_prompt=False
        ) for f in features]

        # 2) 이미지 리스트 (없으면 None)
        images = [f.get("image") for f in features]
        has_image = any(img is not None for img in images)

        # 3) 배치 단위로 processor 호출 → 모델이 원하는 pixel_values / image_grid_thw 생성
        if has_image:
            batch = processor(text=texts, images=images, padding=True, return_tensors="pt")
        else:
            batch = processor(text=texts, padding=True, return_tensors="pt")

        # 4) labels 생성 (pad는 -100)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        # 5) 평가/추론용 id가 있으면 포함
        ids = [f.get("id") for f in features if "id" in f]
        if ids:
            batch["id"] = ids

        return batch

    # Qwen2.5-VL 전용 클래스 사용
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        token=hf_token,
        quantization_config=quantization_config,  # 4bit 로딩 시
        device_map="auto",
        attn_implementation="flash_attention_2", # 추가
        trust_remote_code=True
    )
    
    # QLoRA 준비 (LayerNorm 캐스트, grad 등 세팅) + 체크포인팅과 호환
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    task_dir = os.path.join(config.BASE_OUTPUT_DIR, f"final_adapter_{task}")
    
    training_args = TrainingArguments(
        output_dir=task_dir,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        optim=config.OPTIM,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        logging_steps=config.LOGGING_STEPS,
        save_strategy="no",
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers= os.cpu_count() // 5, # 0
        dataloader_pin_memory=True,
    )
    
    train_dataset = MultitaskDataset(task_df, processor, is_train=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=vl_collate_fn
    )
    
    trainer.train()
    
    model.save_pretrained(task_dir)
    print(f"### Finished training and saved adapter for '{task}' at: {task_dir} ###")

    del model, base_model, processor, trainer, train_dataset, task_df
    gc.collect()
    torch.cuda.empty_cache()
    
    return task_dir   

# --- 4. 어댑터 병합 함수 ---
def merge_adapters(config: Config):
    print("\n" + "="*80)
    print("### STARTING ADAPTER MERGING ###")
    print("="*80)

    gc.collect()
    torch.cuda.empty_cache()
    
    # 병합 시에는 bf16으로 로드하여 정밀도 손실 최소화
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    first_task = config.TASKS[0]
    first_adapter_path = os.path.join(config.BASE_OUTPUT_DIR, f"final_adapter_{first_task}")
    if not os.path.exists(first_adapter_path):
        raise FileNotFoundError(f"First adapter not found at {first_adapter_path}")
        
    peft_model = PeftModel.from_pretrained(base_model, first_adapter_path, adapter_name='default')
    print(f"Loaded base adapter: '{first_task}'")

    for task in config.TASKS[1:]:
        adapter_path = os.path.join(config.BASE_OUTPUT_DIR, f"final_adapter_{task}")
        if os.path.exists(adapter_path):
            peft_model.load_adapter(adapter_path, adapter_name=task)
            print(f"Added adapter: '{task}'")

    # 모든 어댑터 이름을 가져와서 병합
    adapter_names_for_merging = list(peft_model.peft_config.keys())
    
    # PEFT 라이브러리의 add_weighted_adapter 또는 다른 병합 전략 사용 가능
    # 여기서는 가중치를 동일하게 하여 평균내는 방식을 사용
    peft_model.add_weighted_adapter(
        adapters=adapter_names_for_merging,
        weights=[1.0/len(adapter_names_for_merging)] * len(adapter_names_for_merging),
        adapter_name="merged_adapter",
        combination_type="cat"
    )
    
    print(f"\nMerging adapters {adapter_names_for_merging}...")
    peft_model.set_adapter("merged_adapter")
    peft_model.save_pretrained(config.MERGED_ADAPTER_PATH)
    print(f"### Merged adapter saved to: {config.MERGED_ADAPTER_PATH} ###")

    del base_model, peft_model
    gc.collect()
    torch.cuda.empty_cache()

# --- 5. 추론 함수 ---
def run_inference(config: Config):
    print("\n" + "="*80)
    print("### STARTING INFERENCE WITH MERGED ADAPTER ###")
    print("="*80)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # for Low Memory
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        token=hf_token,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(config.MODEL_ID, token=hf_token, trust_remote_code=True)

    if not os.path.exists(config.MERGED_ADAPTER_PATH):
        raise FileNotFoundError(f"Merged adapter not found at {config.MERGED_ADAPTER_PATH}")

    model = PeftModel.from_pretrained(base_model, config.MERGED_ADAPTER_PATH)
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "temperature"):
        model.generation_config.temperature = None
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("### Successfully loaded base model and merged adapter.")

    # 1..N ID 생성
    test_df = pd.read_parquet(config.TEST_DATA_PATH).reset_index(drop=True)
    test_df["id"] = range(0, len(test_df))

    # 메시지/이미지 빌드(학습 때와 동일한 규칙)
    def _load_image_maybe(image_input: str):
        try:
            if isinstance(image_input, str) and image_input.startswith("http"):
                headers = {'User-Agent': 'Mozilla/5.0'}
                r = requests.get(image_input, timeout=15, headers=headers, stream=True)
                r.raise_for_status()
                return Image.open(r.raw).convert("RGB")
            elif isinstance(image_input, str):
                # base64 가정
                return Image.open(BytesIO(base64.b64decode(image_input))).convert("RGB")
        except Exception:
            return None
        return None

    header = ["id", "output"]
    with open(config.SUBMISSION_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(
            f,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writerow(header)

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating predictions (stream)"):
            sample = row.to_dict()
            sid = int(sample["id"])

            try:
                instruction = get_task_instruction(sample)

                image = None
                if sample.get("input_type") == "image":
                    image = _load_image_maybe(sample.get("input", ""))

                # Qwen2.5-VL chat 템플릿 구성
                if image is not None:
                    messages = [
                        {"role": "user", "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": instruction},
                        ]}
                    ]
                else:
                    messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]

                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                # processor로 텐서화 (이미지 유/무에 따라 분기)
                if image is not None:
                    batch = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
                else:
                    batch = processor(text=[prompt], return_tensors="pt", padding=True)

                batch = {k: v.to(device) for k, v in batch.items()}

                gen_kwargs = dict(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask", None),
                    pixel_values=batch.get("pixel_values", None),
                    image_grid_thw=batch.get("image_grid_thw", None),  # ← 추가
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
                # None 값 제거(에러/경고 방지)
                gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

                with torch.no_grad():
                    gen_ids = model.generate(**gen_kwargs)

                # 프롬프트 길이만큼 잘라서 "응답만" 디코드
                prompt_len = batch["input_ids"].shape[1]
                out_text = processor.batch_decode(gen_ids[:, prompt_len:], skip_special_tokens=True)[0]
                out_text = (out_text or "").strip()
                print(f"label {sid}: {out_text}")

                # 스트리밍 기록 (그대로 기록)
                writer.writerow([sid, out_text])

            except Exception as e:
                print(f"[WARN] ID {sid} inference failed: {e}")
                # 실패해도 행은 반드시 기록
                writer.writerow([sid, ""])

    print(f"\n### Submission file created successfully at '{config.SUBMISSION_FILE}' ###")

# --- 6. 메인 실행 함수 ---
def main():
    cfg = Config()
    
    # 실행할 단계를 선택 ('all', 'train', 'merge', 'inference')
    EXECUTION_MODE = 'all'

    if EXECUTION_MODE in ['all', 'train']:
        if os.path.exists(cfg.BASE_OUTPUT_DIR):
            print(f"Removing existing results directory: {cfg.BASE_OUTPUT_DIR}")
            shutil.rmtree(cfg.BASE_OUTPUT_DIR)
        os.makedirs(cfg.BASE_OUTPUT_DIR, exist_ok=True)
        
        full_train_df = pd.read_parquet(cfg.TRAIN_DATA_PATH)
        for task in cfg.TASKS:
            train_task_adapter(cfg, task, full_train_df)

    if EXECUTION_MODE in ['all', 'merge']:
        merge_adapters(cfg)

    if EXECUTION_MODE in ['all', 'inference']:
        run_inference(cfg)

if __name__ == '__main__':
    main()