from typing import Any, Dict
import json
import os
from datetime import datetime

class DataPipeline:
    def __init__(self):
        self.data_dir = "data/pipeline"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def save_agent_output(self, agent_name: str, data: Dict[str, Any]):
        """에이전트의 출력을 JSON 파일로 저장"""
        file_path = f"{self.data_dir}/{agent_name}_output.json"
        # datetime 객체를 문자열로 변환
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, datetime):
                    data[key] = value.isoformat()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_agent_output(self, agent_name: str) -> Dict[str, Any]:
        """저장된 에이전트의 출력을 불러옴"""
        file_path = f"{self.data_dir}/{agent_name}_output.json"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise Exception(f"No output found for agent: {agent_name}")
            
    def clear_agent_output(self, agent_name: str):
        """에이전트의 출력 파일을 삭제"""
        file_path = f"{self.data_dir}/{agent_name}_output.json"
        if os.path.exists(file_path):
            os.remove(file_path) 