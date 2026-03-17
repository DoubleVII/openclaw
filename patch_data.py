import pandas as pd

def patch_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    reward_model_row = []
    for i, row in df.iterrows():
        extra_info = row['extra_info']
        reward_model_row.append({"style": "rule", "ground_truth": extra_info["trg_text"]})

    df["reward_model"] = reward_model_row

    df.to_parquet(path, index=False)


if __name__ == "__main__":
    import fire
    fire.Fire(patch_data)