import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import gzip
import io
from tqdm import tqdm
import multiprocessing as mp
import shutil
import glob

# --- パラメータ設定 ---
# ★★★ ステップ1で取得したご自身のHFトークンをここに設定 ★★★
# 環境変数から読み込むのがより安全ですが、直接記述しても動作します。
HF_TOKEN = os.getenv("HF_TOKEN", "") 

OUTPUT_FILE_PATH = "/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/dataset/c4/c4.jsonl"
NUM_PROCESSES = 64
BASE_URL = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.{i:05d}-of-01024.json.gz"
NUM_FILES = 1024
TMP_DIR = "./c4_tmp"
# --------------------

def create_retry_session():
    """
    リトライ機能を持つrequests.Sessionオブジェクトを作成する関数
    """
    session = requests.Session()
    # ★★★ リトライ戦略の定義 ★★★
    retry_strategy = Retry(
        total=5,  # 合計5回までリトライ
        backoff_factor=5,  # 待機時間（秒）: {backoff factor} * (2 ** ({number of total retries} - 1))
                           # 例: 0s, 2s, 4s, 8s, 16s
        status_forcelist=[429, 500, 502, 503, 504],  # これらのステータスコードでリトライ
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def download_and_write_to_temp(file_index):
    file_url = BASE_URL.format(i=file_index)
    output_filename = os.path.join(TMP_DIR, f"part-{file_index:05d}.jsonl")
    
    session = create_retry_session()
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    
    try:
        # ★★★ session.getを使い、認証ヘッダーを追加 ★★★
        with session.get(file_url, stream=True, timeout=60, headers=headers) as r, open(output_filename, "w", encoding="utf-8") as f_out:
            r.raise_for_status()
            fobj = io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw), encoding='utf-8')
            for line in fobj:
                f_out.write(line)
        return file_index
    except requests.exceptions.RequestException as e:
        print(f"ファイル {file_index} のダウンロードに失敗しました（リトライ上限到達）: {e}")
        return None

# main関数は前回の「解決策2」のままでOK
def main():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.makedirs(TMP_DIR, exist_ok=True)

    with mp.Pool(processes=NUM_PROCESSES) as pool:
        with tqdm(total=NUM_FILES, desc="Downloading parts") as pbar:
            for result in pool.imap_unordered(download_and_write_to_temp, range(NUM_FILES)):
                if result is not None:
                    pbar.update(1)

    print("\nすべてのパーツのダウンロードが完了しました。ファイルを結合します...")
    
    # 正常にダウンロードできたファイルのみを結合対象とする
    temp_files = sorted(glob.glob(os.path.join(TMP_DIR, "*.jsonl")))
    print(f"{len(temp_files)} / {NUM_FILES} 個のファイルを結合します。")

    with open(OUTPUT_FILE_PATH, "wb") as f_out:
        for temp_file in tqdm(temp_files, desc="Merging files"):
            with open(temp_file, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)
    
    print("一時ファイルを削除します...")
    shutil.rmtree(TMP_DIR)

    print(f"\n処理が完了しました。データセットが {OUTPUT_FILE_PATH} に保存されました。")

if __name__ == '__main__':
    main()