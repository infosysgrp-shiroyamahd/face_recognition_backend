from flask import Flask, request, jsonify, send_from_directory
from scipy.spatial import distance as dist
import dlib
import cv2
import insightface
import numpy as np
import psycopg2
import json
from PIL import Image
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from io import BytesIO
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_proto=1)



def connect_to_face_db():
    """
    facedb
    """
    return psycopg2.connect(
        dbname='facedb',
        user='',
        password='',
        host=''
    )

def connect_to_attendance_db():
    """
    kintai
    """
    return psycopg2.connect(
        dbname='kintai',
        user='',
        password='',
        host=''
    )

def connect_to_timecard_db():
    """
    timecard
    """
    return psycopg2.connect(
        dbname='timecard',
        user='',
        password='',
        host=''
    )


def get_shop_name_by_ip(client_ip: str) -> str:
    """
    IP確認
    """
    splitted = client_ip.split(".")
    if len(splitted) < 3:
        return ''

    partial_ip = ".".join(splitted[:3]) + "."

    try:
        conn_timecard = connect_to_timecard_db()
        cursor_time = conn_timecard.cursor()
        cursor_time.execute(
            """
            SELECT shop_name
            FROM shop_ip
            WHERE ip = %s
              AND dflag = 0
            LIMIT 1
            """,
            (partial_ip,)
        )
        result = cursor_time.fetchone()
        cursor_time.close()
        conn_timecard.close()
        return result[0] if result else ''
    except Exception as e:
        print("Error fetching shop_name:", e)
        return ''




def insert_insightface_data(employee_number, employee_name, image_path, face_data):
    """
   TEXT embedding
    """
    conn = connect_to_face_db()
    cursor = conn.cursor()
    for face in face_data:
        face_locations_str = json.dumps(face["bbox"])
        face_landmarks_str = json.dumps(face["landmarks"])
        face_descriptors_str = json.dumps(face["embedding"])
        cursor.execute(
            """
            INSERT INTO insightface_faces
            (employee_number, employee_name, image_path, face_locations, face_landmarks, face_descriptors)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (employee_number, employee_name, image_path,
             face_locations_str, face_landmarks_str, face_descriptors_str)
        )
    conn.commit()
    cursor.close()
    conn.close()


def detect_faces(image_path):
    """
    InsightFace
    """
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=0)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("画像を読み込めません: {}".format(image_path))
    faces = model.get(img)
    return [{
        "bbox": face.bbox.astype(np.float64).tolist(),
        "landmarks": face.kps.astype(np.float64).tolist(),
        "embedding": face.embedding.tolist(),
    } for face in faces]


def find_cosine_similarity(vector_a, vector_b):
    vector_b = vector_b.ravel()
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


@app.route('/api/get_employee_name', methods=['GET'])
def get_employee_name():
    """
    名前確認
    """
    employee_number = request.args.get('employeeNumber', '').strip()
    if not employee_number:
        return jsonify({"employeeName": None}), 200

    try:
        conn = connect_to_attendance_db()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT charge_nm 
            FROM shain_mst
            WHERE charge_cd = %s
            LIMIT 1
            """,
            (employee_number,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            return jsonify({"employeeName": row[0]})
        else:
            return jsonify({"employeeName": None})
    except Exception as e:
        print("Error in get_employee_name:", e)
        return jsonify({"employeeName": None}), 200



@app.route('/api/debug', methods=['GET'])
def debug():
    x_forwarded_for = request.headers.get('X-Forwarded-For')
    if x_forwarded_for:
        client_ip = x_forwarded_for.split(',')[0].strip()
    else:
        client_ip = request.remote_addr
    return jsonify({
        "client_ip": client_ip,
        "x_forwarded_for": x_forwarded_for,
        "remote_addr": request.remote_addr
    })


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'ファイルが選択されていません'}), 400

    image = request.files['image']
    employee_number = request.form.get('employeeNumber', '').strip()
    employee_name = request.form.get('employeeName', '').strip()

    if not employee_number or not employee_name:
        return jsonify({'error': '社員番号と名前を記入してください'}), 400

    conn_face = connect_to_face_db()
    cursor_face = conn_face.cursor()
    cursor_face.execute("SELECT 1 FROM insightface_faces WHERE employee_number = %s", (employee_number,))
    if cursor_face.fetchone() is not None:
        conn_face.close()
        return jsonify({'error': '社員番号既に存在します'}), 409

    filename = secure_filename(employee_number + '.png')
    image_path = os.path.join('uploads', filename)
    image.save(image_path)

    try:
        face_data = detect_faces(image_path)
        if not face_data:
            conn_face.close()
            return jsonify({'message': '顔を検出されていません'}), 200

        cursor_face.execute("SELECT employee_number, employee_name, face_descriptors FROM insightface_faces")
        existing_faces = cursor_face.fetchall()

        potential_matches = []
        for record in existing_faces:
            db_employee_number, db_employee_name, db_face_descriptors_str = record
            db_face_descriptors = np.array(json.loads(db_face_descriptors_str))
            for face in face_data:
                uploaded_descriptor = face["embedding"]
                similarity = find_cosine_similarity(uploaded_descriptor, db_face_descriptors)
                if similarity > 0.6:
                    potential_matches.append({
                        'employee_number': db_employee_number,
                        'employee_name': db_employee_name,
                        'similarity': similarity
                    })

        if potential_matches:
            conn_face.close()
            return jsonify({
                'similar': True,
                'message': '類似な顔が存在します',
                'matches': potential_matches
            }), 200

        insert_insightface_data(employee_number, employee_name, image_path, face_data)
        conn_face.close()
        return jsonify({'message': '保存完了しました'}), 200

    except Exception as e:
        conn_face.close()
        return jsonify({'error': str(e)}), 500


@app.route('/api/re_register', methods=['POST'])
def re_register():
    if 'image' not in request.files:
        return jsonify({'error': 'ファイルが選択されていません'}), 400

    image = request.files['image']
    employee_number = request.form.get('employeeNumber', '').strip()
    employee_name = request.form.get('employeeName', '').strip()

    if not employee_number or not employee_name:
        return jsonify({'error': '社員番号と名前を記入してください'}), 400

    conn_face = connect_to_face_db()
    cursor_face = conn_face.cursor()
    cursor_face.execute("SELECT 1 FROM insightface_faces WHERE employee_number = %s", (employee_number,))
    existing_data = cursor_face.fetchone()
    if not existing_data:
        conn_face.close()
        return jsonify({'error': '社員番号が存在しません。まず顔登録を行ってください。'}), 404

    override = request.form.get('override', 'false') == 'true'
    if not override:
        conn_face.close()
        return jsonify({'error': '社員番号既に存在します。上書きしますか？', 'override_needed': True}), 409

    filename = secure_filename(employee_number + '.png')
    image_path = os.path.join('uploads', filename)
    image.save(image_path)

    try:
        face_data = detect_faces(image_path)
        if not face_data:
            conn_face.close()
            return jsonify({'message': '顔を検出されていません'}), 200

        cursor_face.execute("DELETE FROM insightface_faces WHERE employee_number = %s", (employee_number,))
        conn_face.commit()


        insert_insightface_data(employee_number, employee_name, image_path, face_data)
        conn_face.close()
        return jsonify({'message': '再登録が成功しました'}), 200

    except Exception as e:
        conn_face.close()
        return jsonify({'error': str(e)}), 500



@app.route('/api/verify', methods=['POST'])
def verify():
    if 'image' not in request.files:
        return jsonify({'error': 'ファイルが不足しています'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    image_path = os.path.join('uploads', filename)
    file.save(image_path)

    face_data = detect_faces(image_path)
    if not face_data:
        return jsonify({'message': '顔を検出されていません。'}), 200

    conn_face = connect_to_face_db()
    cursor_face = conn_face.cursor()
    cursor_face.execute("SELECT employee_number, employee_name, face_descriptors FROM insightface_faces")
    all_faces = cursor_face.fetchall()

    for record in all_faces:
        db_employee_number, db_employee_name, db_face_descriptors_str = record
        db_face_descriptors = np.array(json.loads(db_face_descriptors_str))
        for face in face_data:
            uploaded_descriptor = face["embedding"]
            similarity = find_cosine_similarity(uploaded_descriptor, db_face_descriptors)
            if similarity > 0.6:
                conn_face.close()
                return jsonify({
                    'message': f'顔が見つかりました: {db_employee_name} ({db_employee_number})',
                    'similarity': similarity
                }), 200

    conn_face.close()
    return jsonify({'message': '一致する顔が見つかりませんでした。'}), 200



@app.route('/api/liveness', methods=['POST'])
def liveness():
    try:
        frames = []
        for i in range(10):
            file = request.files.get(f'frame{i}')
            if not file:
                return jsonify(message="Not enough frames"), 400
            frame_img = Image.open(BytesIO(file.read()))
            frames.append(frame_img)

        movements = [detect_head_movement(img) for img in frames]
        if check_head_movement_sequence(movements):
            return jsonify(message="Liveness confirmed"), 200
        else:
            return jsonify(message="No valid head movement detected, try again"), 400

    except Exception as e:
        print("Error during POST /liveness:", e)
        return jsonify(error=str(e)), 500

def detect_head_movement(img):
    open_cv_image = np.array(img)[:, :, ::-1].copy()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(open_cv_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return (x + w//2, y + h//2)

def check_head_movement_sequence(movements):

    if not movements or len([m for m in movements if m]) < 2:
        return False
    displacements = []
    prev = None
    for m in movements:
        if m and prev:
            dx = abs(m[0] - prev[0])
            dy = abs(m[1] - prev[1])
            displacements.append(dx + dy)
        prev = m
    avg_disp = sum(displacements) / len(displacements) if displacements else 0
    return avg_disp > 20



@app.route('/api/record_attendance', methods=['POST'])
def record_attendance():
    """
    顔確認
    """
    if 'image' not in request.files or 'status' not in request.form:
        return jsonify({'error': '必要な画像または状態データが不足しています'}), 400

    file = request.files['image']
    status = request.form['status']
    override = request.form.get('override', 'false') == 'true'
    device_time_str = request.form.get('device_time')
    device_time = datetime.fromisoformat(device_time_str) if device_time_str else None

    filename = secure_filename(file.filename)
    image_path = os.path.join('uploads', filename)
    file.save(image_path)

    x_forwarded_for = request.headers.get('X-Forwarded-For')
    client_ip = x_forwarded_for.split(',')[0].strip() if x_forwarded_for else request.remote_addr
    print(f"Client IP: {client_ip}")
    print(f"Device Time: {device_time}")


    conn_face = connect_to_face_db()
    cursor_face = conn_face.cursor()
    faces = detect_faces(image_path)

    cursor_face.execute("SELECT employee_number, employee_name, face_descriptors FROM insightface_faces")
    all_faces = cursor_face.fetchall()
    conn_face.close()

    matched_employee = None
    matched_employee_name = None

    for face in faces:
        uploaded_descriptor = np.array(face['embedding'])
        for record in all_faces:
            db_employee_number, db_employee_name, db_face_descriptors_str = record
            db_face_descriptors = np.array(json.loads(db_face_descriptors_str))
            similarity = find_cosine_similarity(uploaded_descriptor, db_face_descriptors)
            if similarity > 0.6:
                matched_employee = db_employee_number  # sno
                matched_employee_name = db_employee_name  # name
                break
        if matched_employee:
            break

    if not matched_employee:
        return jsonify({'error': '一致する社員が見つかりませんでした'}), 400

    wplace_value = get_shop_name_by_ip(client_ip)

    conn_timecard = connect_to_timecard_db()
    cursor_time = conn_timecard.cursor()

    if status == '1':
        kbn_value = '1'
    elif status == '2':
        kbn_value = '4'
    else:
        kbn_value = status

    current_date = datetime.now().strftime("%Y-%m-%d")
    cursor_time.execute(
        """
        SELECT wdate
        FROM tc_trn_test
        WHERE sno = %s AND kbn = %s AND wdate::date = %s
        """,
        (matched_employee, kbn_value, current_date)
    )
    existing_record = cursor_time.fetchone()

    dflag_value = 0
    update_value = None

    if existing_record and not override:
        cursor_time.close()
        conn_timecard.close()
        return jsonify({
            'override': True,
            'attendance_time': existing_record[0].strftime("%Y-%m-%d %H:%M:%S"),
            'status': status
        }), 200

    if existing_record and override:
        cursor_time.execute(
            """
            DELETE FROM tc_trn_test
            WHERE sno = %s AND kbn = %s AND wdate::date = %s
            """,
            (matched_employee, kbn_value, current_date)
        )
        dflag_value = 1
        update_value = datetime.now()

    wdate_value = datetime.now()

    cursor_time.execute(
        """
        INSERT INTO tc_trn_test
        (sno, name, kbn, wplace, wip, dflag, wdate, pc_date, tid, download, "update")
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            matched_employee,  # sno
            matched_employee_name,  # name
            kbn_value,  # kbn
            wplace_value,  # wplace
            client_ip,  # wip
            dflag_value,  # dflag
            wdate_value,  # wdate
            device_time,  # pc_date
            '01',  # tid
            0,  # download
            update_value  # "update"
        )
    )
    conn_timecard.commit()
    cursor_time.close()
    conn_timecard.close()

    return jsonify({
        'message': '打刻記録が成功しました',
        'employee_number': matched_employee,
        'employee_name': matched_employee_name,
        'attendance_time': wdate_value.strftime("%Y-%m-%d %H:%M:%S"),
        'device_time': device_time_str,
        'wplace': wplace_value,
        'status': status,
        'kbn': kbn_value,
        'dflag': dflag_value,
        'update': update_value.strftime("%Y-%m-%d %H:%M:%S") if update_value else None
    }), 200



if __name__ == '__main__':
    ssl_cert_path = '/etc/ssl/cert.pem'
    ssl_key_path = '/etc/ssl/key.pem'
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=(ssl_cert_path, ssl_key_path))
