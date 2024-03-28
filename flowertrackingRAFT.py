import os.path
from raftalg import Raftalg
import torch
import time
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from RAFTsort import RFsort
from RAFTsort import Tracker1, Tracker2, Tracker3


plot = [[6900, 7152]]


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='v8xbest.pt')
    parser.add_argument('--image_resize', default=640, type=int)
    parser.add_argument('--det_conf_thresh', default=0.5, type=float)
    parser.add_argument('--det_iou_thresh', default=0.3, type=float)
    parser.add_argument('--sort_max_age', default=30, type=int)
    parser.add_argument('--sort_min_hit', default=2, type=int)
    parser.add_argument('--data', default="20220907103911428290")
    parser.add_argument('--imgpath', default="./img/")
    parser.add_argument('--inde', default=26, type=int)
    parser.add_argument('--savepath', default="./results/")
    parser.add_argument('--save_text', default=True)
    return parser.parse_args()


class YOLOv8:
    def __init__(self, weight, device, imgsz=640):
        self.weight = weight
        self.device = device
        self.imgsz = imgsz
        self.model = YOLO(weight)

    def detectimg(self, img, conf, iou):
        results = self.model.predict(source=img, conf=conf, iou=iou)
        return results[0].boxes.boxes.to('cpu').numpy()


if __name__ == "__main__":
    args = parse_args()
    save_text = args.save_text
    data = args.data
    imgpath = args.imgpath
    inde = args.inde
    save_path = args.savepath
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for plo in plot:
        for i1 in range(plo[0], plo[0] + 1):
            fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
            out1 = cv2.VideoWriter(os.path.join(save_path, data + 'v1_' + str(inde) + '.avi'),
                                   fourcc, 30.0, (640, 480))
            out2 = cv2.VideoWriter(os.path.join(save_path, data + 'v2_' + str(inde) + '.avi'),
                                   fourcc, 30.0, (640, 480))
            out3 = cv2.VideoWriter(os.path.join(save_path, data + 'v3_' + str(inde) + '.avi'),
                                   fourcc, 30.0, (640, 480))
            if save_text == True:
                textsavepath = os.path.join(save_path, 'results' + str(inde) + '.txt')
                with open(textsavepath, 'a+') as f:
                    f.write('frame,cam,ID,xmin,ymin,xmax,ymax' + '\n')

            bbox = []
            count = 0
            frame_id = 0
            result12 = []
            result22 = []
            result32 = []
            trackers1 = []
            trackers2 = []
            trackers3 = []

            # Initialize Detector and Tracker
            Detector = YOLOv8(weight=args.weights, device='0', imgsz=args.image_resize)
            mot_tracker1 = RFsort(args.sort_max_age, args.sort_min_hit)
            mot_tracker2 = RFsort(args.sort_max_age, args.sort_min_hit)
            mot_tracker3 = RFsort(args.sort_max_age, args.sort_min_hit)
            raft = Raftalg('./RAFT/raft-things.pth', 'cuda')

            colours = np.random.rand(32, 3) * 255
            img11 = cv2.imread(imgpath + 'color_img3_' + data + '/frame' + str(i1) + '.jpg')
            img21 = cv2.imread(imgpath + 'color_img2_' + data + '/frame' + str(i1) + '.jpg')
            img31 = cv2.imread(imgpath + 'color_img1_' + data + '/frame' + str(i1) + '.jpg')
            # Get the detection results
            result1 = Detector.detectimg(img11, args.det_conf_thresh, args.det_iou_thresh)
            result2 = Detector.detectimg(img21, args.det_conf_thresh, args.det_iou_thresh)
            result3 = Detector.detectimg(img31, args.det_conf_thresh, args.det_iou_thresh)

            pre_img11 = cv2.cvtColor(img11, cv2.COLOR_BGR2RGB)
            pre_img11 = torch.from_numpy(pre_img11).permute(2, 0, 1).float()
            pre_img11 = pre_img11[None].to('cuda')

            pre_img21 = cv2.cvtColor(img21, cv2.COLOR_BGR2RGB)
            pre_img21 = torch.from_numpy(pre_img21).permute(2, 0, 1).float()
            pre_img21 = pre_img21[None].to('cuda')

            pre_img31 = cv2.cvtColor(img31, cv2.COLOR_BGR2RGB)
            pre_img31 = torch.from_numpy(pre_img31).permute(2, 0, 1).float()
            pre_img31 = pre_img31[None].to('cuda')

            all_pts1 = {}
            all_pts2 = {}
            all_pts3 = {}
            dep1 = {}
            dep2 = {}
            dep3 = {}
            for i1 in range(plo[0] + 1, plo[1]):
                # Read RGB and depth images
                img12 = cv2.imread(imgpath + 'color_img3_' + data + '/frame' + str(i1) + '.jpg')
                img22 = cv2.imread(imgpath + 'color_img2_' + data + '/frame' + str(i1) + '.jpg')
                img32 = cv2.imread(imgpath + 'color_img1_' + data + '/frame' + str(i1) + '.jpg')

                depth12 = cv2.imread(imgpath + 'depth_img3_' + data + '/' + str(i1) + '.png', -1)
                depth22 = cv2.imread(imgpath + 'depth_img2_' + data + '/' + str(i1) + '.png', -1)
                depth32 = cv2.imread(imgpath + 'depth_img1_' + data + '/' + str(i1) + '.png', -1)

                im12 = img12.copy()
                im22 = img22.copy()
                im32 = img32.copy()

                img1track = img12.copy()
                img2track = img22.copy()
                img3track = img32.copy()

                s = time.time()
                result12 = Detector.detectimg(img12, args.det_conf_thresh, args.det_iou_thresh)
                result22 = Detector.detectimg(img22, args.det_conf_thresh, args.det_iou_thresh)
                result32 = Detector.detectimg(img32, args.det_conf_thresh, args.det_iou_thresh)

                current_img12 = cv2.cvtColor(img12, cv2.COLOR_BGR2RGB)
                current_img12 = torch.from_numpy(current_img12).permute(2, 0, 1).float()
                current_img12 = current_img12[None].to('cuda')

                current_img22 = cv2.cvtColor(img22, cv2.COLOR_BGR2RGB)
                current_img22 = torch.from_numpy(current_img22).permute(2, 0, 1).float()
                current_img22 = current_img22[None].to('cuda')

                current_img32 = cv2.cvtColor(img32, cv2.COLOR_BGR2RGB)
                current_img32 = torch.from_numpy(current_img32).permute(2, 0, 1).float()
                current_img32 = current_img32[None].to('cuda')

                # Calcualte the dense optical flow
                flow_up1 = raft.calculateopticflow(pre_img11, current_img12)
                flo1 = flow_up1[0].permute(1, 2, 0).cpu().numpy()
                u1 = flo1[:, :, 0]
                v1 = flo1[:, :, 1]

                flow_up2 = raft.calculateopticflow(pre_img21, current_img22)
                flo2 = flow_up2[0].permute(1, 2, 0).cpu().numpy()
                u2 = flo2[:, :, 0]
                v2 = flo2[:, :, 1]

                flow_up3 = raft.calculateopticflow(pre_img31, current_img32)
                flo3 = flow_up3[0].permute(1, 2, 0).cpu().numpy()
                u3 = flo3[:, :, 0]
                v3 = flo3[:, :, 1]

                # Process the detection and tracking results
                det1 = result12[:, 0:5]
                # Get the tracking results
                trackers1, trackerboxes1 = mot_tracker1.update(det1, flo1, 1)
                # Show detection results
                for d in det1:
                    cv2.putText(im12, '%.2f' % d[4], (int(d[2]) - 10, int(d[3]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1)

                det2 = result22[:, 0:5]
                trackers2, trackerboxes2 = mot_tracker2.update(det2, flo2, 2)
                for d in det2:
                    cv2.putText(im22, '%.2f' % d[4], (int(d[2]) - 10, int(d[3]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1)

                det3 = result32[:, 0:5]
                trackers3, trackerboxes3 = mot_tracker3.update(det3, flo3, 3)
                for d in det3:
                    cv2.putText(im32, '%.2f' % d[4], (int(d[2]) - 10, int(d[3]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1)

                keep_line_idx1 = []
                keep_line_idx2 = []
                keep_line_idx3 = []
                # Visualize the tracking results
                if len(trackers1) != 0:
                    for p in trackers1:
                        depth = []
                        xmin = int(p[0])
                        ymin = int(p[1])
                        xmax = int(p[2])
                        ymax = int(p[3])
                        label = int(p[4])
                        keep_line_idx1.append(label)
                        if label in all_pts1:
                            all_pts1[label].append(((xmin + xmax) // 2, (ymin + ymax) // 2))
                        else:
                            all_pts1[label] = [((xmin + xmax) // 2, (ymin + ymax) // 2)]
                        crop = img12[ymin:ymax, xmin:xmax]
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        threshold, img_bin = cv2.threshold(crop, -1, 255, cv2.THRESH_OTSU)
                        positions = np.where(img_bin == 255)
                        for i in range(len(positions[0])):
                            x = xmin + positions[0][i]
                            y = ymin + positions[1][i]
                            print(x, y)
                            if x < 640 and y < 480:
                                if depth12[y, x] != 0:
                                    depth.append(depth12[y, x])
                        dep1[label] = np.mean(depth)
                        cv2.putText(im12, 'F%d' % p[4], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                                    2)
                        cv2.rectangle(im12, (xmin, ymin), (xmax, ymax), (
                            int(colours[label % 32, 0]), int(colours[label % 32, 1]), int(colours[label % 32, 2])), 3)

                        cv2.putText(img1track, 'F%d' % p[4], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2)
                        cv2.rectangle(img1track, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)

                        if save_text == True:
                            with open(textsavepath, 'a+') as f:
                                # frame, cam, id, bbox
                                item = str(i1 - plo[0]) + ',1,' + str(p[4]) + ',' + str(xmin) + ',' + str(
                                    ymin) + ',' + str(xmax) + ',' + str(ymax) + '\n'
                                f.write(item)

                if len(trackers2) != 0:
                    for p in trackers2:
                        depth = []
                        xmin = int(p[0])
                        ymin = int(p[1])
                        xmax = int(p[2])
                        ymax = int(p[3])
                        label = int(p[4])
                        keep_line_idx2.append(label)
                        if label in all_pts2:
                            all_pts2[label].append(((xmin + xmax) // 2, (ymin + ymax) // 2))
                        else:
                            all_pts2[label] = [((xmin + xmax) // 2, (ymin + ymax) // 2)]

                        crop = img22[ymin:ymax, xmin:xmax]
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        threshold, img_bin = cv2.threshold(crop, -1, 255, cv2.THRESH_OTSU)
                        cv2.imshow('bin', img_bin)

                        positions = np.where(img_bin == 255)
                        for i in range(len(positions[0])):
                            x = xmin + positions[0][i]
                            y = ymin + positions[1][i]
                            print(x, y)
                            if x < 640 and y < 480:
                                if depth22[y, x] != 0:
                                    depth.append(depth22[y, x])
                        dep2[label] = np.mean(depth)

                        cv2.putText(im22, 'F%d' % p[4], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                                    2)
                        cv2.rectangle(im22, (xmin, ymin), (xmax, ymax), (
                            int(colours[label % 32, 0]), int(colours[label % 32, 1]), int(colours[label % 32, 2])), 3)

                        cv2.putText(img2track, 'F%d' % p[4], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
                        cv2.rectangle(img2track, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)

                        if save_text == True:
                            with open(textsavepath, 'a+') as f:
                                # frame, cam, id, bbox
                                item = str(i1 - plo[0]) + ',2,' + str(p[4]) + ',' + str(xmin) + ',' + str(
                                    ymin) + ',' + str(xmax) + ',' + str(ymax) + '\n'
                                f.write(item)

                if len(trackers3) != 0:
                    for p in trackers3:
                        depth = []
                        xmin = int(p[0])
                        ymin = int(p[1])
                        xmax = int(p[2])
                        ymax = int(p[3])
                        label = int(p[4])
                        keep_line_idx3.append(label)
                        if label in all_pts3:
                            all_pts3[label].append(((xmin + xmax) // 2, (ymin + ymax) // 2))
                        else:
                            all_pts3[label] = [((xmin + xmax) // 2, (ymin + ymax) // 2)]

                        crop = img32[ymin:ymax, xmin:xmax]
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        threshold, img_bin = cv2.threshold(crop, -1, 255, cv2.THRESH_OTSU)
                        positions = np.where(img_bin == 255)
                        for i in range(len(positions[0])):
                            x = xmin + positions[0][i]
                            y = ymin + positions[1][i]
                            print(x, y)
                            if x < 640 and y < 480:
                                if depth32[y, x] != 0:
                                    depth.append(depth32[y, x])
                        dep3[label] = np.mean(depth)

                        cv2.putText(im32, 'F%d' % p[4], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                                    2)
                        cv2.rectangle(im32, (xmin, ymin), (xmax, ymax), (
                            int(colours[label % 32, 0]), int(colours[label % 32, 1]), int(colours[label % 32, 2])), 3)

                        cv2.putText(img3track, 'F%d' % p[4], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 153, 255), 2)
                        cv2.rectangle(img3track, (xmin, ymin), (xmax, ymax), (0, 153, 255), 3)

                        if save_text == True:
                            with open(textsavepath, 'a+') as f:
                                # frame, cam, id, bbox
                                item = str(i1 - plo[0]) + ',3,' + str(p[4]) + ',' + str(xmin) + ',' + str(
                                    ymin) + ',' + str(xmax) + ',' + str(ymax) + '\n'
                                f.write(item)

                fps = 1. / float(time.time() - s)
                cv2.putText(im12, 'FPS: {:.1f} flowers: {} detection: {}'.format(fps, len(all_pts1), len(result12)),
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.imshow("Tracking1", im12)
                # cv2.imshow("Tracking1", img1track)

                cv2.putText(im22, 'FPS: {:.1f} flowers: {} detection: {}'.format(fps, len(all_pts2), len(result22)),
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.imshow("Tracking2", im22)
                # cv2.imshow("Tracking2", img2track)

                cv2.putText(im32, 'FPS: {:.1f} flowers: {} detection: {}'.format(fps, len(all_pts3), len(result32)),
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.imshow("Tracking3", im32)
                # cv2.imshow("Tracking3", img3track)

                if cv2.waitKey(1) & 0xff == 27:
                    break
                out1.write(im12)
                pre_img11 = current_img12

                out2.write(im22)
                pre_img21 = current_img22

                out3.write(im32)
                pre_img31 = current_img32

            else:
                Tracker1.count = 0
                Tracker2.count = 0
                Tracker3.count = 0
                inde += 1
                out1.release()
                out2.release()
                out3.release()
                break
    cv2.destroyAllWindows()
