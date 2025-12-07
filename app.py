import os
import json
import re
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)

app = Flask(__name__)
app.jinja_env.variable_start_string = '[['
app.jinja_env.variable_end_string = ']]'

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- MODEL CONFIG ---
# Prompt hệ thống cực kỳ nghiêm ngặt về việc tuân thủ dữ liệu
system_prompt = """
Bạn là Trợ lý Jrai. Nhiệm vụ của bạn là dịch thuật và giải thích tiếng Jrai (Gia Lai).

QUY TẮC TUYỆT ĐỐI:
1. ƯU TIÊN DỮ LIỆU ĐƯỢC CUNG CẤP: Nếu người dùng hỏi về một từ có trong phần "THÔNG TIN TỪ ĐIỂN" mà tôi cung cấp kèm theo, bạn PHẢI dùng định nghĩa đó. Không được dùng kiến thức bên ngoài nếu nó mâu thuẫn.
2. CHÍNH XÁC: Ví dụ, nếu từ điển nói "Chào" là "Kơkuh", thì bạn phải trả lời là "Kơkuh", không được trả lời là "Hiam" hay từ khác.
3. KHÔNG BỊA ĐẶT: Nếu không tìm thấy từ trong ngữ cảnh và không chắc chắn, hãy nói "Tôi chưa có dữ liệu chính xác về từ này trong từ điển".
"""

generation_config = {
    "temperature": 0.3, # Giảm sáng tạo để tăng độ chính xác
    "top_p": 0.95,
    "max_output_tokens": 8192,
}

try:
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-preview-09-2025",
        generation_config=generation_config,
        system_instruction=system_prompt
    )
except:
    model = None

# --- HELPER FUNCTIONS ---
def read_json(filename):
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except: return []

def save_json(filename, data):
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except: pass

def search_context(query):
    """
    Hàm tìm kiếm từ khóa trong từ điển JSON để mớm cho AI.
    Giúp AI trả lời chính xác theo dữ liệu local.
    """
    vocab = read_json('vocabulary.json')
    alphabet = read_json('alphabets.json')
    query = query.lower()
    
    found_info = []
    
    # 1. Quét từ điển
    for item in vocab:
        # Kiểm tra nếu từ khoá xuất hiện trong tiếng Việt hoặc Jrai
        jrai = item.get('jrai', '').lower()
        viet = item.get('viet', '').lower()
        
        if jrai in query or viet in query:
            info = f"- Từ: {item['jrai']} | Nghĩa: {item['viet']} | Loại: {item.get('type','')} | Ví dụ: {item.get('example','')}"
            found_info.append(info)
            
    # 2. Quét bảng chữ cái (nếu người dùng hỏi về phát âm)
    for item in alphabet:
        char = item.get('char', '').lower()
        if f"chữ {char}" in query or f"âm {char}" in query:
             info = f"- Chữ cái: {item['char']} | Phát âm: {item['pronounce']} | Ví dụ: {item['example']}"
             found_info.append(info)

    if found_info:
        return "THÔNG TIN TỪ ĐIỂN TÌM THẤY (HÃY DÙNG THÔNG TIN NÀY ĐỂ TRẢ LỜI):\n" + "\n".join(found_info)
    return ""

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_all_data():
    return jsonify({
        "alphabet": read_json('alphabets.json'),
        "dictionary": read_json('vocabulary.json'),
        "lessons": read_json('lessons.json'),
        "user": read_json('user.json'),
        "library": {
            "text": read_json('library_text.json'),
            "audio": read_json('library_audio.json')
        }
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    if not model: return jsonify({"error": "AI Error"}), 500

    user_input = request.json.get('message', '')
    history = request.json.get('history', [])
    
    # BƯỚC QUAN TRỌNG: Tìm dữ liệu trong JSON trước
    context_data = search_context(user_input)
    
    # Ghép dữ liệu tìm được vào câu hỏi để bắt buộc AI dùng
    full_prompt = ""
    if context_data:
        full_prompt = f"{context_data}\n\nCâu hỏi của người dùng: {user_input}"
    else:
        full_prompt = user_input

    try:
        # Gửi kèm lịch sử để hội thoại liền mạch
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(full_prompt)
        return jsonify({"response": response.text})
    except Exception as e:
        print(f"Lỗi: {e}")
        return jsonify({"error": "Lỗi kết nối AI"}), 500

@app.route('/api/user/update', methods=['POST'])
def update_user():
    # Code cập nhật user giữ nguyên như cũ
    user_data = read_json('user.json')
    req_data = request.form
    if 'name' in req_data: user_data['name'] = req_data['name']
    if 'email' in req_data: user_data['email'] = req_data['email']
    if 'phone' in req_data: user_data['phone'] = req_data['phone']
    if 'theme' in req_data: user_data['theme'] = req_data['theme']
    
    if 'avatar' in request.files:
        file = request.files['avatar']
        if file.filename:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            user_data['avatar'] = f"/static/uploads/{filename}"

    save_json('user.json', user_data)
    return jsonify({"status": "success", "user": user_data})

if __name__ == '__main__':
    app.run(debug=True, port=5000)