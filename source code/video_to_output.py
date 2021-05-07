'''
비디오에서 인체의 키포인트를 분석하여 그 결과를 이미지와 텍스트 파일로 저장하는 코드이다.
--file로 원하는 비디오를 지정한다. 지정하지 않으면 자동으로 컴퓨터 내장 카메라를 사용한다.
기본적으로 이미지는 output_data라는 이름의 디렉토리에 저장하도록 만들었기 때문에 사전에 이 디렉토리를 만들어 놔야한다.
텍스트 파일은 output.txt로 저장되면 --output_text를 통해 이름을 바꿔 저장 가능하다. 
'''

import tensorflow as tf
import cv2 # 이미지 사용
import time
import argparse
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
parser.add_argument('--output_text', type=str, default='output1.txt')
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
        
        # 파일에 결과 저장
        sys.stdout = open(args.output_text, 'w')

        start = time.time()
        frame_count = 0
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
            
            # 이미지 저장
            cv2.imwrite('C:/Users/98dms/capstone1/posenet_python_final/output1_data/' + image_name, overlay_image)


            # 결과 데이터 텍스트로 저장
            print()
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                       break
                 #print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    print(s, str(c).strip('[]'), end = ' ')

            # 이미지에 텍스트 추가
            #cv2.putText(overlay_image, "I PT", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))

            cv2.imshow('I PT', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()
