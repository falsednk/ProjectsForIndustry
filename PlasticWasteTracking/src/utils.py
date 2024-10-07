import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import motmetrics as mm

def imshow_frame(inp, norm=None, title=None, plt_ax=plt,  mode=None, color = None, pil=None):
    if mode=='tensor':
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([*norm[0]])
        std = np.array([*norm[-1]])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt_ax.imshow(inp)
    else:
        if pil is None:
            plt_ax.imshow(inp, cmap=color)
        else:
            inp_RGB = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            plt_ax.imshow(inp_RGB, cmap=color)
    try:
        if title is not None:
            plt_ax.set_title(title)
    except:
        plt.title(title)
    plt_ax.grid(False)

def convert_xywh_xyxy(bbox):
    x, y, w, h = bbox
    return [int(x), int(y), int(x+w), int(y+w)]

def convert_xywh_xywh(bbox):
    x, y, w, h = bbox
    return [x-w/2, y-h/2, w, h]

def convert_xyxy_xywh(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2-x1, y2-y1]

def show_track(frame, result, width, time, track_history):
    
    color_map = {i: tuple(map(lambda x: x*255, sns.color_palette('pastel', 15)[-(i+1)])) for i in range(15)}
    for box, track_id, cls in zip(result['boxes'], result['track_id'], result['class']):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x+(w-x)/2), float(y+(h-y)/2)))
        
        if len(track) > 30:  
            track.pop(0)
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        
        cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=12)
        cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), color_map[int(cls)], 8)
        cv2.putText(frame, f'id:{str(track_id)} class:{str(int(cls))}', (int(x) + 20, int(y) - 40), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(230, 230, 230), thickness=8)
    cv2.putText(frame, f'processing time:{time:.2f} msec', (width-800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=(200, 50, 50), thickness=6)
    imshow_frame(frame)
    return track_history

def get_info_frame(result):
    results_frame = {'rf': []}
    for box, track_id, cls in zip(result['boxes'], result['track_id'], result['class']):
        x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
        results_frame['rf'].append({
                'bbox': [x, y, w, h], 
                'track_id': track_id, 
                'class': int(cls)
            })
    return results_frame      

def show_track_for_segmentation(frame, bboxs, obj_id, masks, track_history, labels_tracks, time=None):
    
    color_map = {i: tuple(map(lambda x: x*255, sns.color_palette('pastel', 20)[-(i+1)])) for i in range(20)}
    for box, track_id in zip(bboxs, obj_id):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x+(w-x)/2), float(y+(h-y)/2)))
        cls = labels_tracks[track_id]
        if len(track) > 30:  
            track.pop(0)
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        
        cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=12)
        cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), color_map[int(cls)], 8)
        cv2.putText(frame, f'id:{str(track_id)} class:{str(int(cls))}', (int(x) + 20, int(y) - 40), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(230, 230, 230), thickness=8)
        
    cv2.putText(frame, f'processing time:{time:.2f} msec', (width-800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=(200, 50, 50), thickness=6)
    plt.imshow(frame)
    
    for mask, track_id in zip(masks, obj_id):
        show_mask(mask, plt.gca(), obj_id=track_id)

    return track_history

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap('tab10', 30)
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    ax.imshow(mask_image)

def bbox_iou_np(boxes1, boxes2):
    x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    box1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    box2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    iou = inter_area / (box1_area[:, None] + box2_area - inter_area)

    return iou


def get_info_frame_sam2(bboxs, obj_id, labels_tracks):
    results_frame = {'rf': []}
    for box, track_id in zip(bboxs, obj_id):
        x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
        results_frame['rf'].append({
                'bbox': [x, y, w, h], 
                'track_id': track_id, 
                'class': int(labels_tracks[track_id])
            })
    return results_frame    

def get_tg_info(row):
    x, y, w, h = row['bbox']
    return x, y, w, h 

def get_metrics_tracker(gt, tg, fr_min, fr_max, max_iou=1):
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in range(fr_min, fr_max+1):
        gt_dets = gt[gt[:,0]==frame,1:6] 
        t_dets = tg[tg[:,0]==frame,1:6]
        C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], max_iou=max_iou)
        acc.update(gt_dets[:,0].astype('int').tolist(), t_dets[:,0].astype('int').tolist(), C)
            
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['recall', 'precision',  'mota', 'motp'], name='acc')
    return summary

def get_metrics_tracker_in_detail(gt, tg, fr_min, fr_max, max_iou=1):
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in range(fr_min, fr_max+1):
        gt_dets = gt[gt[:,0]==frame,1:6] 
        t_dets = tg[tg[:,0]==frame,1:6]
        C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], max_iou=max_iou)
        acc.update(gt_dets[:,0].astype('int').tolist(), t_dets[:,0].astype('int').tolist(), C)
            
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                     'recall', 'precision', 'num_objects', \
                                     'mostly_tracked', 'partially_tracked', \
                                     'mostly_lost', 'num_false_positives', \
                                     'num_misses', 'num_switches', \
                                     'num_fragmentations', 'mota', 'motp' \
                                    ], name='acc')
    return summary, acc

def write_new_file(input_file, output_file):
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    return cap, output    
