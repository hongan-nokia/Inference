import huggingface_hub
from huggingface_hub import snapshot_download

huggingface_hub.login("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", add_to_git_credential=True)

taskList = [
    "task1391_winogrande_easy_answer_generation",
    "task290_tellmewhy_question_answerability",
    "task1598_nyc_long_text_generation",
    "task620_ohsumed_medical_subject_headings_answer_generation",
]

for task in taskList:
    snapshot_download(
        repo_id=f"Lots-of-LoRAs/{task}",
        repo_type="dataset",
        local_dir=f"./{task}",
        cache_dir=None,
        proxies={
            "https": "http://10.144.1.10:8080",
            "http": "http://10.144.1.10:8080"
        },
        max_workers=3,
        resume_download=True,
    )
