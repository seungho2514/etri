import sys
import os

import sys
import os

def setup_paths():
    # 프로젝트 루트 경로 (/workspace/etri)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    libs_path = os.path.join(root, "libs")
    
    if os.path.exists(libs_path):
        # 1. libs 폴더 자체 추가
        if libs_path not in sys.path:
            sys.path.append(libs_path)
        
        # 2. libs 내의 모든 하위 디렉토리를 탐색하여 sys.path에 추가
        for root_dir, dirs, files in os.walk(libs_path):
            for directory in dirs:
                full_path = os.path.join(root_dir, directory)
                if full_path not in sys.path:
                    sys.path.append(full_path)
                    
        print(f"✅ Python paths updated for all folders in libs/")
    else:
        print(f"⚠️ Warning: libs folder not found at {libs_path}")