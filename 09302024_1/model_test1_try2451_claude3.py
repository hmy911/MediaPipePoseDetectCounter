import cv2
import os
import itertools
import copy
import csv
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tqdm import tqdm

# Mediapipe Pose 模块初始化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def relative_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y, base_z = 0, 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

    shoulder_length_x = abs(temp_landmark_list[11][0] - temp_landmark_list[12][0])

    if shoulder_length_x == 0:
        shoulder_length_x = 1

    for idx, relative_point in enumerate(temp_landmark_list):
        temp_landmark_list[idx][0] = temp_landmark_list[idx][0] / shoulder_length_x
        temp_landmark_list[idx][1] = temp_landmark_list[idx][1] / shoulder_length_x
        temp_landmark_list[idx][2] = temp_landmark_list[idx][2] / shoulder_length_x

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    return temp_landmark_list

def play_video_with_landmarks(video_path, model, dframe, sumframe):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        relative_landmark_list_total = []
        class_name_mapping = {
            0: "pushup", 1: "abworkout", 2: "squat", 3: "pullup",
            4: "run", 5: "jump", 9: "rest",
        }
        frame_count = 0 
        predicted_class = 9  # 初始化为休息状态
        predicted_class_count = {}  # 统计每个类别的预测次数
        predicted_class_max = ""  # 初始化最大预测次数的类别
        predicted_class_max_count = 0  # 初始化最大预测次数
        unit_frame_count = 0  # 初始化 30 帧计数器

        # 初始化计数器
        counters = {class_name: 0 for class_name in class_name_mapping.values()}

        # 创建输出目录
        output_dir = 'output_images'
        os.makedirs(output_dir, exist_ok=True)

        # 创建CSV文件
        csv_path = 'output_data.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['frame', 'second', 'predicted_class_max', 'pushup', 'abworkout', 'squat', 'pullup', 'run', 'jump', 'rest', 'totalcounter'])

        pbar = tqdm(total=total_frames, desc="Processing frames")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # 绘制骨骼关键点
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                post_landmark_list = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
                relative_landmark_list = relative_landmark(post_landmark_list)
                relative_landmark_list_total.append(relative_landmark_list)
                
                if len(relative_landmark_list_total) >= dframe:
                    input_data = np.array(relative_landmark_list_total[-dframe:]).reshape(1, dframe, -1)
                    predictions = model.predict(input_data, verbose=0)
                    predicted_class = np.argmax(predictions, axis=-1)[0]
                    predicted_class_name = class_name_mapping.get(predicted_class, "未知")

                    print(f"预测类别: {predicted_class_name}")  # 打印预测类别

                    # 更新预测次数统计
                    predicted_class_count[predicted_class_name] = predicted_class_count.get(predicted_class_name, 0) + 1

                    # 在画面上显示预测结果
                    cv2.putText(frame, f'Predicting: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # 显示预测概率
                    for i, (class_name, prob) in enumerate(zip(class_name_mapping.values(), predictions[0])):
                        cv2.putText(frame, f"{class_name}: {prob:.2f}", (10, 60 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # 统计 sumframe 帧
                    unit_frame_count += 1
                    if unit_frame_count == sumframe:
                        predicted_class_max = max(predicted_class_count, key=predicted_class_count.get)
                        predicted_class_max_count = predicted_class_count[predicted_class_max]
                        counters[predicted_class_max] += 1
                        unit_frame_count = 0
                        predicted_class_count = {}

                    cv2.putText(frame, f"Most predicted: {predicted_class_max}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 显示帧数和FPS
            frame_count += 1
            second = frame_count / fps
            cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'FPS: {fps:.2f}', (frame.shape[1] - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 保存图像
            output_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(output_path, frame)

            # 更新CSV文件
            total_counter = sum(counters.values())
            with open(csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([frame_count, second, predicted_class_max] + list(counters.values()) + [total_counter])

            cv2.imshow('Video with Landmarks', frame)
            pbar.update(1)

            frame_count += 1
            # if(frame_count > 500):
            #     break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        pbar.close()

# 使用示例
video_path = 'video/edit/demoA4.mp4' 
model_path = 'my_lstm_model_1.h5'  # 模型1
model = load_model(model_path)
play_video_with_landmarks(video_path, model, 1, 30)  # 假设 dframe 为 1，sumframe 为 30


