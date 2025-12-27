import requests
import json
import time
import sys

def test_minference_server(model_name="Qwen2.5-7B-Instruct-1M", url="http://localhost:8000/v1/chat/completions"):
    print(f"--- Testing Connection to {url} ---")
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": "Explain what sparse attention is in one sentence."} 
        ],
        "temperature": 0,
        "max_tokens": 100
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    try:
        start_time = time.time()
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=60)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
            
            print("\n[Success!]")
            print(f"Response Time: {duration:.2f} seconds")
            print(f"Model Answer: {answer}")
            print(f"Tokens Used: {usage}")
        else:
            print(f"\n[Error] Status Code: {response.status_code}")
            print(f"Response Body: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n[Error] Could not connect to the server. Is it running on port 8000?")
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred: {e}")

if __name__ == "__main__":
    # 如果运行脚本时带了参数，则第一个参数视为模型名称
    target_model = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-7B-Instruct-1M"
    test_minference_server(target_model)
