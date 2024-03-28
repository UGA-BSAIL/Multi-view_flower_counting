import cv2
import numpy as np

# Extrisnic parameters
rotation12 = np.array([[-0.022120807638068, 0.997001113252052, 0.074158276973035],
                       [-0.838196828180352, -0.058925076691309, 0.542175167787424],
                       [0.544919028019611, -0.050165879949356, 0.836986641106696]])
rotation32 = np.array([[-0.007469169975655, -0.999910857745887, -0.011067432482709],
                       [0.881052317230586, -0.011815253327206, 0.472871244728869],
                       [-0.472959856438673, -0.006219031331204, 0.881062028376442]])
translation12 = np.array([-7.532220117316310e+02, 36.429476660754180, 3.269824727891138e+02])
translation32 = np.array([8.363944735199757e+02, 1.516889922088871e+02, 3.683182872570054e+02])

# Intrisnic parameters
cam1 = [605.327, 605.419, 320.132, 249.355]
cam2 = [602.936, 602.915, 317.666, 243.243]
cam3 = [606.33, 606.314, 320.146, 237.015]
cam2intri = np.array([[cam2[0], 0, cam2[2]], [0, cam2[1], cam2[3]], [0, 0, 1]])

plot = [[6900, 7152]]


def getxyxy(data):
    xmin, ymin, xmax, ymax = data[4], data[5], data[6], data[7]
    return [xmin, ymin, xmax, ymax]


def segment(rgbimg, depthimg, position):
    depth = []
    xmin, ymin, xmax, ymax = position[0], position[1], position[2], position[3]
    crop = rgbimg[ymin:ymax, xmin:xmax]
    cv2.imshow("corp", crop)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    threshold, img_bin = cv2.threshold(crop, -1, 255, cv2.THRESH_OTSU)
    cv2.imshow("seg", img_bin)
    cv2.waitKey(1)
    positions = np.where(img_bin == 255)
    for i in range(len(positions[0])):
        x = xmin + positions[1][i]
        y = ymin + positions[0][i]
        if x < 640 and y < 480:
            if depthimg[y, x] != 0:
                depth.append(depthimg[y, x])
    if depth == []:
        return 0
    else:
        return np.median(depth)


def xyxy2xy(location):
    x = location[0] + (location[2] - location[0]) / 2
    y = location[1] + (location[3] - location[1]) / 2
    return x, y


def get3D(cam, pos2d, depth):
    if cam == 1:
        z = float(depth)
        x = (pos2d[0] - cam1[2]) * z / cam1[0]
        y = (pos2d[1] - cam1[3]) * z / cam1[1]

    if cam == 2:
        z = float(depth)
        x = (pos2d[0] - cam2[2]) * z / cam2[0]
        y = (pos2d[1] - cam2[3]) * z / cam2[1]

    if cam == 3:
        z = float(depth)
        x = (pos2d[0] - cam3[2]) * z / cam3[0]
        y = (pos2d[1] - cam3[3]) * z / cam3[1]
    return x, y, z


def get_dict_key(dic, value):
    key = list(dic.keys())[list(dic.values()).index(value)]
    return key


if __name__ == '__main__':
    save_txt = True
    dat = '20220907103911428290'
    imgpath = './img/'

    indexx = 26
    for plo in plot:
        trkresults = './results/results' + str(indexx) + '.txt'
        if save_txt == True:
            with open("./transformed2D/new2Dposresult" + str(indexx) + ".txt", 'a+') as f0:
                f0.write("frame,cam,ID,x,y,x2,y2" + '\n')

        with open(trkresults) as f0:
            lines = f0.readlines()
        for line in lines:
            line = line.rstrip()
            if "frame" in line:
                pass
            else:
                f, c, d, x1, y1, x2, y2 = line.split(",")
                if int(c) == 1:
                    cd = 3
                elif int(c) == 2:
                    cd = 2
                elif int(c) == 3:
                    cd = 1
                rgbimg = cv2.imread(
                    imgpath + 'color_img' + str(cd) + '_' + dat + '/frame' + str(int(f) + plo[0]) + '.jpg')
                depthimg = cv2.imread(imgpath + 'depth_img' + str(cd) + '_' + dat + '/' + str(int(f) + plo[0]) + '.png',
                                      -1)
                pos2d = [int(x1), int(y1), int(x2), int(y2)]
                depth = segment(rgbimg, depthimg, pos2d)
                x, y = xyxy2xy(pos2d)
                if depth != 0:
                    if int(c) == 1:
                        pos1_3d = get3D(1, [x, y], depth)
                        pos12_3d = np.dot(np.transpose(rotation12), np.transpose(np.array(pos1_3d))) + np.transpose(
                            translation12)
                        pos12_2d = np.dot(cam2intri, pos12_3d)
                        x2, y2 = pos12_2d[0] / pos12_2d[2], pos12_2d[1] / pos12_2d[2]
                    elif int(c) == 2:
                        x2, y2 = x, y
                    elif int(c) == 3:
                        pos3_3d = get3D(3, [x, y], depth)
                        pos32_3d = np.dot(np.transpose(rotation32), np.transpose(np.array(pos3_3d))) + np.transpose(
                            translation32)
                        pos32_2d = np.dot(cam2intri, pos32_3d)
                        x2, y2 = pos32_2d[0] / pos32_2d[2], pos32_2d[1] / pos32_2d[2]
                    # Remove projections that are out of boundary
                    if y2 >= 0 and y2 <= 480 and save_txt == True:
                        with open("./transformed2D/new2Dposresult" + str(indexx) + ".txt", 'a+') as f1:
                            string_file = f + ',' + c + ',' + d + ',' + str(round(x, 2)) + ',' + str(
                                round(y, 2)) + ',' + str(round(x2, 2)) + ',' + str(round(y2, 2)) + '\n'
                            f1.write(string_file)

        indexx += 1
