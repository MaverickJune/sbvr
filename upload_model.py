from huggingface_hub import HfApi, upload_folder

# 사용자와 업로드할 저장소 경로 지정
repo_id = "appleyu/Qwen_Qwen3-8B_4_16_16_w_rotate_pt"  # 예: "appleyu/Qwen3-8B-sbvr-pt"

# 업로드할 폴더 경로
folder_path = "./quantized_model/Qwen_Qwen3-8B_4_16_16_w_rotate"

# 업로드 (repo_type='model'은 일반 PyTorch 모델일 때 기본값)
upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model"
)
