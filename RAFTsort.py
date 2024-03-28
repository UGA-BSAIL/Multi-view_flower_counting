from __future__ import print_function
import numpy as np
from scipy.optimize import linear_sum_assignment
import copy
import math


def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def distance(bb_det, bb_trk):
    x1 = bb_det[0] + (bb_det[2] - bb_det[0]) / 2
    y1 = bb_det[1] + (bb_det[3] - bb_det[1]) / 2
    x2 = bb_trk[0] + (bb_trk[2] - bb_trk[0]) / 2
    y2 = bb_trk[1] + (bb_trk[3] - bb_trk[1]) / 2
    dis = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    print(dis)
    return dis


def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class Tracker1(object):
    count = 0

    def __init__(self, bbox):
        self.x = bbox[:4]
        self.time_since_update = 0
        self.id = Tracker1.count
        Tracker1.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.u = []
        self.v = []
        self.status = 1

    def update(self, bbox, umean, vmean):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.x[0] = bbox[0]
        self.x[1] = bbox[1]
        self.x[2] = bbox[2]
        self.x[3] = bbox[3]
        self.u.append(umean)
        self.v.append(vmean)
        self.status = 1

    def predict(self, u, v):
        self.x[0] += u
        self.x[1] += v
        self.x[2] += u
        self.x[3] += v

        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x)
        return self.history[-1]

    def get_state(self):
        return self.x


class Tracker2(object):
    count = 0

    def __init__(self, bbox):
        self.x = bbox[:4]
        self.time_since_update = 0
        self.id = Tracker2.count
        Tracker2.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.u = []
        self.v = []
        self.status = 1

    def update(self, bbox, umean, vmean):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.x[0] = bbox[0]
        self.x[1] = bbox[1]
        self.x[2] = bbox[2]
        self.x[3] = bbox[3]
        self.u.append(umean)
        self.v.append(vmean)
        self.status = 1

    def predict(self, u, v):
        self.x[0] += u
        self.x[1] += v
        self.x[2] += u
        self.x[3] += v

        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x)
        return self.history[-1]

    def get_state(self):
        return self.x


class Tracker3(object):
    count = 0

    def __init__(self, bbox):
        self.x = bbox[:4]
        self.time_since_update = 0
        self.id = Tracker3.count
        Tracker3.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.u = []
        self.v = []
        self.status = 1

    def update(self, bbox, umean, vmean):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.x[0] = bbox[0]
        self.x[1] = bbox[1]
        self.x[2] = bbox[2]
        self.x[3] = bbox[3]
        self.u.append(umean)
        self.v.append(vmean)
        self.status = 1

    def predict(self, u, v):
        self.x[0] += u
        self.x[1] += v
        self.x[2] += u
        self.x[3] += v
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x)
        return self.history[-1]

    def get_state(self):
        return self.x


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.1, dis_threshold=50):
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)
    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)
    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if (len(np.array(unmatched_trackers)) == 0 or len(np.array(unmatched_detections)) == 0):
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    dis_matrix = np.zeros((len(np.array(unmatched_detections)), len(np.array(unmatched_trackers))), dtype=np.float32)
    i = 0
    j = 0
    for detind in np.array(unmatched_detections):
        for trkind in np.array(unmatched_trackers):
            dis_matrix[int(i), int(j)] = distance(detections[detind], trackers[trkind])
            j = j + 1
        i = i + 1
        j = 0
    matched_ind = linear_assignment(dis_matrix)
    # print(matched_ind)
    temp_unmatched_detections = copy.deepcopy(unmatched_detections)
    temp_unmatched_trackers = copy.deepcopy(unmatched_trackers)
    for n in matched_ind:
        # print(n)
        # print(dis_matrix[n[0], n[1]])
        if dis_matrix[n[0], n[1]] < dis_threshold:
            match = np.array([[temp_unmatched_detections[n[0]], temp_unmatched_trackers[n[1]]]])
            unmatched_detections.remove(temp_unmatched_detections[n[0]])
            unmatched_trackers.remove(temp_unmatched_trackers[n[1]])
            matches.append(match)
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class RFsort(object):
    def __init__(self, max_age=1, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, flo, num):
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        temp_u = []
        temp_v = []
        for t, trk in enumerate(trks):
            if self.trackers[t].status == 1:
                box = self.trackers[t].x
                u = flo[int(box[1]):int(box[3]), int(box[0]):int(box[2]), 0]
                v = flo[int(box[1]):int(box[3]), int(box[0]):int(box[2]), 1]
                umean = np.mean(u)
                vmean = np.mean(v)
                temp_u.append(umean)
                temp_v.append(vmean)
                pos = self.trackers[t].predict(umean, vmean)
                trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if self.trackers[t].status == 0:
                if self.trackers[t].u != []:
                    umean = np.mean([x for x in self.trackers[t].u if not math.isnan(x)])
                    vmean = np.mean([x for x in self.trackers[t].v if not math.isnan(x)])
                    temp_u.append(umean)
                    temp_v.append(vmean)
                    pos = self.trackers[t].predict(umean, vmean)
                    trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
                else:
                    box = self.trackers[t].x
                    u = flo[int(box[1]):int(box[3]), int(box[0]):int(box[2]), 0]
                    v = flo[int(box[1]):int(box[3]), int(box[0]):int(box[2]), 1]
                    umean = np.mean(u)
                    vmean = np.mean(v)
                    temp_u.append(umean)
                    temp_v.append(vmean)
                    pos = self.trackers[t].predict(umean, vmean)
                    trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        if len(dets) > 0:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

            # update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if (t not in unmatched_trks):
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    # print(d)
                    umean = temp_u[t]
                    vmean = temp_v[t]
                    trk.update(dets[d, :][0], umean, vmean)

            # create and initialise new trackers for unmatched detections
            for i in unmatched_dets:
                if num == 1:
                    trk = Tracker1(dets[i, :])
                    self.trackers.append(trk)
                if num == 2:
                    trk = Tracker2(dets[i, :])
                    self.trackers.append(trk)
                if num == 3:
                    trk = Tracker3(dets[i, :])
                    self.trackers.append(trk)

            for ind, untrk in enumerate(self.trackers):
                if (ind in unmatched_trks):
                    self.trackers[ind].status = 0

            i = len(self.trackers)
            for trk in reversed(self.trackers):
                d = trk.get_state()
                if ((trk.time_since_update < 1) and (
                        trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                    ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
                i -= 1
                # remove dead tracklet
                if (trk.time_since_update > self.max_age):
                    self.trackers.pop(i)
            if (len(ret) > 0):
                return np.concatenate(ret), trks
            return np.empty((0, 5)), trks

        else:
            for ind, untrk in enumerate(self.trackers):
                self.trackers[ind].status = 0
            return np.empty((0, 5)), trks
