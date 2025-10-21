import numpy as np
import sys
import os

def print_npz_info(file_path):
    """
    .npz または .npy(辞書) ファイルを読み込み、その中身の情報を出力する
    """
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません: {file_path}")
        return

    print("=" * 80)
    print(f" inspecting file: {file_path}")
    print("=" * 80)

    try:
        # allow_pickle=True でロード
        data = np.load(file_path, allow_pickle=True)
        
        keys = []
        
        # --- ここから修正 ---
        # data が NpzFile オブジェクトか、それとも dict かを判別
        if hasattr(data, 'files'):
            # ケース1: NpzFile オブジェクトの場合 (.npz アーカイブ)
            keys = data.files
            print(f"[キー一覧 (NpzFile - {len(keys)} 個)]")
        
        elif isinstance(data, dict):
            # ケース2: dict オブジェクトの場合 (辞書が .npy として保存されたもの)
            keys = data.keys()
            print(f"[キー一覧 (dict - {len(keys)} 個)]")

        elif isinstance(data, np.ndarray) and data.dtype == 'object' and isinstance(data.item(), dict):
             # ケース3: 辞書が0次元配列に格納されている場合
             data = data.item() # 辞書を取り出す
             keys = data.keys()
             print(f"[キー一覧 (dict in ndarray - {len(keys)} 個)]")
        
        else:
            # それ以外（単一の .npy 配列など）
            print(f"エラー: 予期しないデータ型です: {type(data)}")
            if hasattr(data, 'close'):
                data.close()
            return
        # --- ここまで修正 ---
            
        print(keys)
        print("-" * 80)

        for key in keys:
            item = data[key]
            
            print(f"Key: '{key}'")
            
            if isinstance(item, np.ndarray):
                print(f"  ├─ Type:  numpy.ndarray")
                print(f"  ├─ Shape: {item.shape}")
                print(f"  ├─ Dtype: {item.dtype}")
                if item.size > 20:
                    print(f"  └─ Data (最初の5件): {item.flatten()[:5]}")
                else:
                    print(f"  └─ Data: {item}")
            else:
                print(f"  ├─ Type:  {type(item)}")
                print(f"  └─ Data:  {item}")
            
            print("-" * 80)

        # NpzFile オブジェクトの場合は close する
        if hasattr(data, 'close'):
            data.close()

    except Exception as e:
        print(f"ファイルの読み込み中にエラーが発生しました: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python check_npz.py <.npz ファイルへのパス>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    print_npz_info(file_path)