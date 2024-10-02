# pip install GPUtil
import cv2
import os
import itertools
import copy
import csv
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time  # For FPS calculation
import GPUtil

# 创建输出目录
output_dir = 'output_frames'
os.makedirs(output_dir, exist_ok=True)

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

def calculate_angle(a, b, c):
    """计算三点形成的角度"""
    a = np.array(a)  # 将列表转换为 NumPy 数组
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

def draw_angle_indicator(frame, point, angle, stage, color):
    h, w, _ = frame.shape
    cx, cy = int(point[0] * w), int(point[1] * h)
    radius = 30
    start_angle = -angle
    end_angle = 0
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_angle, end_angle, color, 2)
    cv2.putText(frame, f"{angle:.1f}", (cx - 10, cy - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, stage.upper(), (cx - 15, cy + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def play_video_with_landmarks(video_path, model, dframe):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return
        relative_landmark_list_total = []
        class_name_mapping = {
            0: "pushup",
            1: "abworkout",
            2: "squat",
            3: "pullup",
            4: "run",
            5: "jump",
            9: "rest",
        }
        # dframe = 30  # 每秒预测的帧数
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
        predicted_class = 9  # 初始化为休息状态
        predicted_class_count = {}  # 统计每个类别的预测次数
        text_y = 150  # 初始化文字起始位置
        predicted_class_max = ""  # 初始化最大预测次数的类别
        predicted_class_max_count = 0  # 初始化最大预测次数
        unit_frame_count = 0  # 初始化 30 帧计数器
        start_time = time.time()  # 记录开始时间
        
        # 获取视频原始帧率
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        # CSV 文件写入准备
        csv_filename = 'output_data.csv'
        csv_file = open(csv_filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'second', 'predicted_class_max', "pushup", "abworkout", "squat", "pullup", "run", "jump", "rest", "totalcounter"])  # 添加 CSV 文件头
        
        # 累加计数器
        pushup_count = 0
        abworkout_count = 0
        squat_count = 0
        pullup_count = 0
        run_count = 0
        jump_count = 0
        rest_count = 0
        total_counter = 0

        # 角度计數和階段
        counter = 0
        stage = "shoulder-down"
        stage2 = "hip-down"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # background blue
            # cv2.rectangle(frame, (0, 0), (230, 800), (245, 117, 16), -1)

            results = pose.process(frame_rgb)
            if results.pose_landmarks and hasattr(results.pose_landmarks, 'landmark'):
                post_landmark_list = []
                for value in results.pose_landmarks.landmark:
                    temp_value = [value.x, value.y, value.z]
                    post_landmark_list.append(temp_value)

                relative_landmark_list = relative_landmark(post_landmark_list)
                relative_landmark_list_total.append(relative_landmark_list)

                # Check if enough frames are accumulated
                if len(relative_landmark_list_total) < dframe:
                    print(f"Not enough frames for prediction ({len(relative_landmark_list_total)} < {dframe})")
                    continue # Skip prediction if not enough frames

                # 每帧进行预测
                input_data = np.array(relative_landmark_list_total[-dframe:]).reshape(1, dframe, -1)
                predictions = model.predict(input_data)
                # print("预测结果:", predictions)  # 打印原始预测结果
                predicted_class = np.argmax(predictions, axis=-1)[0]
                predicted_class_name = class_name_mapping.get(predicted_class, "未知")
                print(f"预测类别: {predicted_class_name}")  # 打印预测类别

                #background blue 创建一个透明的蓝色矩形作为背景
                overlay = frame.copy()  # 创建一个图像副本
                cv2.rectangle(overlay, (0, 0), (200, 150), (125, 0, 0), -1)  # 蓝色矩形
                # cv2.rectangle(overlay, (0, 0), (200, 100), (245, 117, 16), -1)  # 蓝色矩形
                alpha = 0.9  # 透明度，0 为完全透明，1 为完全不透明
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)  # 将透明矩形叠加到原图像


                # 在画面上显示预测结果
                cv2.putText(frame, f'predicting: {predicted_class_name}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # 字变小50%

                # 更新帧数计数器
                frame_count += 1
                cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)  # 字变小50%

                # 更新预测次数统计
                if predicted_class_name in predicted_class_count:
                    predicted_class_count[predicted_class_name] += 1
                else:
                    predicted_class_count[predicted_class_name] = 1

                # 在画面上显示预测结果汇总
                text_y = 100  # 调整汇总信息的起始位置
                for i, (class_name, count) in enumerate(predicted_class_count.items()):
                    cv2.putText(frame, f"{class_name}: {count}/30", (10, text_y + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)  # 字变小50%

                # 统计 30 帧
                unit_frame_count += 1
                if unit_frame_count == 30:
                    # 找出 30 帧内出现次数最多的类别
                    predicted_class_max = max(predicted_class_count, key=predicted_class_count.get)
                    predicted_class_max_count = predicted_class_count[predicted_class_max]

                    # 显示 30 帧内出现次数最多的类别
                    cv2.putText(frame, f"Most predicted: {predicted_class_max}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # 字变大，红色

                    # 更新累加计数器
                    if predicted_class_max == "pushup":
                        pushup_count += 1
                    elif predicted_class_max == "abworkout":
                        abworkout_count += 1
                    elif predicted_class_max == "squat":
                        squat_count += 1
                    elif predicted_class_max == "pullup":
                        pullup_count += 1
                    elif predicted_class_max == "run":
                        run_count += 1
                    elif predicted_class_max == "jump":
                        jump_count += 1
                    elif predicted_class_max == "rest":
                        rest_count += 1
                    total_counter += 1

                    # 写入 CSV 文件
                    current_second = round(frame_count / video_fps)  # 获取当前秒数
                    csv_writer.writerow([frame_count, current_second, predicted_class_max, pushup_count, abworkout_count, squat_count, pullup_count, run_count, jump_count, rest_count, total_counter])

                    # 重置计数器和统计字典
                    unit_frame_count = 0
                    predicted_class_count = {}
                else:
                    cv2.putText(frame, f"Most predicted: {predicted_class_max}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # 字变大，红色
                
                #########################################################################################################################################
                # # 計算角度和計數
                #########################################################################################################################################

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = pose.process(frame)
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                try:
                    landmarks = results.pose_landmarks.landmark
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                    angle = calculate_angle(hip, shoulder, elbow)
                    angleHip = calculate_angle(shoulder, hip, knee)

                    #background blue 创建一个透明的蓝色矩形作为背景
                    overlay = frame.copy()  # 创建一个图像副本
                    cv2.rectangle(overlay, (0, 150), (200, 800), (245, 117, 16), -1)  # 蓝色矩形
                    alpha = 0.9  # 透明度，0 为完全透明，1 为完全不透明
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)  # 将透明矩形叠加到原图像

                    # posupdown= basey + (basegap*5)
                    posupdown = basemy + 45
                    # 判斷肩膀動作
                    cv2.putText(frame, f"degress:", (10, posupdown-15),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

                    if angleHip > 90:
                        if angle > 45:
                            stage = "shoulder-up"
                            # 根據條件設置顏色
                            draw_angle_indicator(frame, shoulder, angle, stage, color=(0, 255, 0))  # 綠色
                            cv2.putText(frame, f"{stage}: {angle:.1f} ", (10, posupdown),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

                        elif angle < 35 and stage == 'shoulder-up':
                            stage = "shoulder-down"
                            counter += 1
                            draw_angle_indicator(frame, shoulder, angle, stage, color=(0, 0, 255))  # 紅色
                            cv2.putText(frame, f"{stage}: {angle:.1f} ", (10, posupdown),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
                    else:
                        # 根據條件設置顏色
                        draw_angle_indicator(frame, shoulder, angle, stage, color=(0, 255, 0))  # 綠色
                        cv2.putText(frame, f"{stage}: {angle:.1f} ", (10, posupdown),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)


                    # 判斷髖部動作
                    if angleHip > 60:
                        stage2 = "hip-up"
                        # 根據條件設置顏色
                        draw_angle_indicator(frame, hip, angleHip, stage2, color=(0, 255, 0))  # 綠色
                        cv2.putText(frame, f"{stage2} degrees:{angleHip:.1f} ", (10, posupdown+15),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

                    elif angleHip < 45 and stage2 == 'hip-up':
                        stage2 = "hip-down"
                        counter += 1
                        # 根據條件設置顏色
                        draw_angle_indicator(frame, hip, angleHip, stage2, color=(0, 0, 255))  # 紅色
                        cv2.putText(frame, f"{stage2}: {angleHip:.1f} ", (10, posupdown+15),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

                    else:
                        # 根據條件設置顏色
                        draw_angle_indicator(frame, hip, angleHip, stage2, color=(0, 255, 0))  # 綠色
                        cv2.putText(frame, f"{stage2}: {angleHip:.1f}", (10, posupdown+15),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                except:
                    pass
              
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                
                #-------------------------------------counter display start ------------------------------------------------#
                #-----------------------------------------------------------------------------------------------------------#
                # 定义每个项目的进度条参数
                basex = 10
                basemy = 180
                basey = basemy+60
                basegap = 35
                base_bar_width = 100
                total_progress_bar_x = basex
                total_progress_bar_y = basemy
                total_progress_bar_width = base_bar_width
                total_progress_bar_height = 10

                pushup_progress_bar_x = basex
                pushup_progress_bar_y = basey+ (basegap*1)
                pushup_progress_bar_width = base_bar_width
                pushup_progress_bar_height = 10

                abworkout_progress_bar_x = basex
                abworkout_progress_bar_y = basey + (basegap*2)
                abworkout_progress_bar_width = base_bar_width
                abworkout_progress_bar_height = 10

                squat_progress_bar_x = basex
                squat_progress_bar_y = basey + (basegap*3)
                squat_progress_bar_width = base_bar_width
                squat_progress_bar_height = 10

                pullup_progress_bar_x = basex
                pullup_progress_bar_y = basey + (basegap*4)
                pullup_progress_bar_width = base_bar_width
                pullup_progress_bar_height = 10

                # 显示每个项目的文字计数
                cv2.putText(frame, f"Total Counter : {total_counter}",
                            (total_progress_bar_x, total_progress_bar_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                cv2.putText(frame, f"Counter Push-up : {pushup_count}",
                            (pushup_progress_bar_x, pushup_progress_bar_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

                cv2.putText(frame, f"Counter Abworkout : {abworkout_count}",
                            (abworkout_progress_bar_x, abworkout_progress_bar_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

                cv2.putText(frame, f"Counter Pull-up : {pullup_count}",
                            (pullup_progress_bar_x, pullup_progress_bar_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

                cv2.putText(frame, f"Counter Squat : {squat_count}",
                            (squat_progress_bar_x, squat_progress_bar_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

                # 绘制每个项目的进度条
                # Total
                cv2.rectangle(frame, (total_progress_bar_x, total_progress_bar_y),
                            (total_progress_bar_x + int(total_progress_bar_width * (total_counter * 0.05)),
                                total_progress_bar_y + total_progress_bar_height),
                            (0, 0, 255), -1)  # 橘色填充

                # Push-ups
                cv2.rectangle(frame, (pushup_progress_bar_x, pushup_progress_bar_y),
                            (pushup_progress_bar_x + int(pushup_progress_bar_width * (pushup_count * .1)),  # 计算填充宽度
                                pushup_progress_bar_y + pushup_progress_bar_height),
                            (0, 255, 0), -1)  # 绿色填充

                # Ab Workouts
                cv2.rectangle(frame, (abworkout_progress_bar_x, abworkout_progress_bar_y),
                            (abworkout_progress_bar_x + int(abworkout_progress_bar_width * (abworkout_count * .1)),
                                abworkout_progress_bar_y + abworkout_progress_bar_height),
                            (0, 255, 0), -1)  # 黄色填充

                # Pull-ups
                cv2.rectangle(frame, (pullup_progress_bar_x, pullup_progress_bar_y),
                            (pullup_progress_bar_x + int(pullup_progress_bar_width * (pullup_count * .1)),
                                pullup_progress_bar_y + pullup_progress_bar_height),
                            (0, 255, 0), -1)  # 红色填充

                # Squats
                cv2.rectangle(frame, (squat_progress_bar_x, squat_progress_bar_y),
                            (squat_progress_bar_x + int(squat_progress_bar_width * (squat_count * .1)),
                                squat_progress_bar_y + squat_progress_bar_height),
                            (0, 255, 0), -1)  # 蓝色填充

                #-------------------------------------counter display end ------------------------------------------------#

                # 绘制关键点
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                base2y = 500
                # 计算并显示帧率
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                start_time = end_time
                cv2.putText(frame, f"Video FPS: {int(video_fps)} Runtime FPS: {int(fps)}", (10, base2y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)  # 字变小50%

                # 顯示GPU名稱
                gpus = GPUtil.getGPUs()
                gpu_name = gpus[0].name  # 获取第一个 GPU 设备的名称
                cv2.putText(frame, f"GPU: {gpu_name}", (10, base2y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

                # 添加进度条
                progress_bar_width = 150
                progress_bar_height = 2  # 进度条高度变小
                progress_bar_x = 10
                progress_bar_y = base2y+10
                bar_width = int(progress_bar_width * frame_count / total_frames)
                cv2.rectangle(frame, (progress_bar_x, progress_bar_y), (progress_bar_x + progress_bar_width, progress_bar_y + progress_bar_height), (0, 255, 0), 1)
                cv2.rectangle(frame, (progress_bar_x, progress_bar_y), (progress_bar_x + bar_width, progress_bar_y + progress_bar_height), (0, 255, 0), -1)

            # 顯示圖片
            cv2.imshow('Video with Landmarks', frame)
            # 保存图像
            output_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(output_path, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # 关闭 CSV 文件
        csv_file.close()

# 使用示例
video_path = 'video/edit/demoA9.mp4' 
model_path = 'my_lstm_model_1.h5' #模型1
model = load_model(model_path)
play_video_with_landmarks(video_path, model, 30) 

'''

'''
