import tensorflow as tf
import cv2 # 이미지 사용
import time
import argparse
import numpy as np # 배열 사용
import math # 각도 구할 때 사용
import sys # 파일 저장을 위해
import os # 파일 저장을 위해

import posenet

# Argument를 파싱한다.
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file) # 지정한 파일을 캡쳐
        else:
            cap = cv2.VideoCapture(args.cam_id) # 실시간 카메라 캡쳐
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        image_num = 0 # 이미지에 번호 부여

        start = time.time()
        frame_count = 0
        good_count = 0
        down = 0
        up = 0
        print('good_count : ', good_count)
        while True: # 반복해서 이미지 출력 -> 여기서 계속 비교가 들어가야 할듯!!

            image_num = image_num + 1 # 계속해서 번호 증가
            image_name = str(image_num) + '.jpg'
            
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            res = np.zeros((17,3)) # 결과를 저장하는 행렬 생성
            angle1 = 0
            angle2 = 0
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                       break
                #print()
                #print('#pi %d' % pi)
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                   # print(ki)
                    #print(s, str(c).strip('[]'))
                    #print('res');
                    if s > 0.4:
                        res[ki, 0] = s
                        res[ki, 1] = c[0]
                        res[ki, 2] = c[1]
                        x1 = res[15, 1]
                        x2 = res[13, 1]
                        y1 = res[15, 2]
                        y2 = res[13, 2]
            
                        width1 = math.fabs(x1 - x2)
                        height1 = math.fabs(y1 - y2)
                        if width1 == 0:
                            angle1 = -1
                        else:
                            angle1 = height1 / width1

                        p1 = res[16, 1]
                        p2 = res[14, 1]
                        k1 = res[16, 2]
                        k2 = res[14, 2]
                        width2 = math.fabs(p1 - p2)
                        height2 = math.fabs(k1 - k2)
                        if width2 == 0:
                            angle2 = -1
                        else:
                            angle2 = height2 / width2

                    #print(res)
            # res배열에 키포인트 데이터 저장 완료
            # 여기서부터 각도, 비율 구해서 배열 저장
            
            # 기존 바른 자세 데이터와 비교

            cv2.putText(overlay_image, 'angle1: '+str(angle1), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            cv2.putText(overlay_image, 'angle2: '+str(angle2), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

            if (0.35 < angle1 < 0.45) and (0.42 < angle2 < 0.52):
                cv2.putText(overlay_image, 'GOOD!!!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))
                down = 1

            if (0.14 < angle1 < 0.24) and (0.05 < angle2 < 0.15):
                if down == 1 :
                    good_count = good_count +1
                    down = 0
                    print('good_count : ', good_count)

            #print('good_count : ', good_count)
            # 횟수 카운트, 피트백을 해야하는데

            # 이미지에 텍스트 추가
            #cv2.putText(overlay_image, 'nose x: '+str(res[0, 1]), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            # 비교해서 맞으면 출력(x좌표 기준 반대라는 것 유의!!)

            #if 500. < res[0, 1] < 600.:
             #   cv2.putText(overlay_image, 'GOOD!!!!', (300, 330), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))

            
            
        #print('Average FPS: ', frame_count / (time.time() - start))

            # 이 위에서 이미지 편집은 끝내야함!!
            # 영상(이미지)을 뿌리는 코드
                        
            cv2.imshow('I PT', overlay_image)
            # 이미지 저장
            #cv2.imwrite('C:/Users/owner/Documents/2019 Capstone/posenet_python/output_data/' + image_name, overlay_image)
            #이미지 주소 자신의 컴퓨터에 맞추기
            cv2.imwrite('C:/Users/98dms/capstone1/posenet_python_final/output2_data/' + image_name, overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))
        

if __name__ == "__main__":
    main()
