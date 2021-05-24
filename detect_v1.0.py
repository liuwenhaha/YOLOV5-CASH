"""
20210419:
python3 detect_v1.0.py --weights best_exp6_996.pt --source data/videos/58_cut.avi --conf-thres 0.45
"""

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import logging
import pdb

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger('detect_v1.0')


# 函数定义

# 获得当前摄像头电子围栏信息
# 比如，根据摄像头id、场景名等
def get_fence(cam_id, scene):
    
    # 调用格林API获取当前摄像头电子围栏，并解析为如下格式
    #
    # ......
    #

    # 测试数据
    fences = [
        [470, 200, 1220, 1070],
        [1230, 200 , 1840, 1070]
    ]
    return fences    


# 获取当前帧所有'bulk cash', 'staff'类别预测框坐标
def get_boxes(det):
    cash_boxes, staff_boxes = [], []

    # 每个框判断，属于staff_boxes 或 cash_boxes 
    for *xyxy, conf, cls in reversed(det):
        # xyxy is a tensor list, [tensor,tensor,tensor,tensor]
        # if cash
        if int(cls) == 0:
            cash_boxes.append([int(x.cpu()) for x in xyxy])  #从GPU转入CPU
        # if staff
        if int(cls) == 1:
            staff_boxes.append([int(x.cpu()) for x in xyxy])
    return cash_boxes, staff_boxes


# 框与电子围栏相交判断，统计电子围栏相交框数
def box_inter(fence, boxes):
    [x1, y1, x2, y2]  = fence
    count = 0
    for box in boxes:
        [a1, b1, a2, b2] = box
        # 两个矩形框是否相交判断
        if (x2 > a1 and x1 < a2 and y1 < b2 and y2 > b1):
            count += 1 # 如果相交，相交box数加一
    return count  

# 违规放置大量现金判断：无人且有钱
def cash_illegal(cash_boxes_in, staff_boxes_in):

    # 单帧是否无人且有钱
    if staff_boxes_in==0 and cash_boxes_in!=0:
        return True
    else:
        return False

# 非阻塞固定长度队列，存储历史连续N帧违规判别结果
def puter(l, data):
    max_ = max_save_frame
    if len(l) == max_:
        l.pop(0)
    l.append(data)

    
# 绘图展示    
def show_image(det, im0, names, colors, fence_color, imgbox, frame):

    # 调用原代码，实现预测框绘制
    for *xyxy, conf, cls in reversed(det):
        # Add bbox to image
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
    

    # 增加：电子围栏绘制、WARN告警信息显示
    tl = round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize('fencex, sum=30', 0, fontScale=tl / 3, thickness=tf)[0]
    for i, fence in enumerate(fences):
        # fence_box
        c1, c2 = (int(fence[0]), int(fence[1])), (int(fence[2]), int(fence[3]))
        cv2.rectangle(im0, c1, c2, fence_color, thickness=tl, lineType=cv2.LINE_AA)
        # fence_text
        c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
        cv2.rectangle(im0, c1, c2, fence_color, -1, cv2.LINE_AA)  # filled
        text = 'fence{}, sum={}'.format(i, sum(fence_result[i]))
        cv2.putText(im0, text, (c1[0], c1[1] + t_size[1]), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # warn
        if sum(fence_result[i]) / max_save_frame > abnormal_thres:
            c1 = c1[0], c1[1] + 2*t_size[1] + 3
            c2 = c1[0] + t_size[0], c1[1] + 2*t_size[1] + 3
            cv2.rectangle(im0, c1, c2, (48,48,255), -1, cv2.LINE_AA)  # filled
            cv2.putText(im0, 'CASH WARN!', (c1[0], c1[1] + 2*t_size[1] - 5), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
         
    # 显示其他信息：如帧号
    cv2.putText(im0, 'frame: {}'.format(frame), (10, 30), \
                0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    # py展示
    #cv2.imshow('x', im0)
    # jupyter展示
    #imgbox.value = cv2.imencode('.jpg', im0)[1].tobytes() 
    
    return im0

# 修改原检测函数，加入现金违规后处理
def detect():

    # jupyter展示
    ### create
    #imgbox = widgets.Image(format='jpg', height=600, width=600)
    #display(imgbox)

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    ## Second-stage classifier
    #classify = False
    #if classify:
    #    modelc = load_classifier(name='resnet101', n=2)  # initialize
    #    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    fence_color = [random.randint(0, 255) for _ in range(3)]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    frame = 0
    for path, img, im0s, vid_cap in dataset:
        frame += 1

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        """
        前向传播 返回pred的shape是(1, num_boxes, 5+num_class)
        前向传播 返回pred的shape是(1, num_boxes, 5+num_class)
        w为传入网络图片的长和宽，注意dataset在检测时使用了矩形推理，所以这里h不一定等于w
        num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
        pred[..., 0:4]为预测框坐标
        预测框坐标为xywh(中心点+宽长)格式
        pred[..., 4]为objectness置信度
        pred[..., 5:-1]为分类结果
        """
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        #pdb.set_trace()

        # Apply NMS
        """
        pred:前向传播的输出
        conf_thres:置信度阈值
        iou_thres:iou阈值
        classes:是否只保留特定的类别
        agnostic:进行nms是否也去除不同类别之间的框
        经过nms之后，预测框格式：xywh-->xyxy(左上角右下角)
        pred是一个列表list[torch.tensor]，长度为batch_size
        每一个torch.tensor的shape为(num_boxes, 6),内容为box+conf+cls
        """
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        
        # pred格式:[batch_size, N, 6]
        #pdb.set_trace()

        ## Apply Classifier
        #if classify:
        #    pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                ## --------------------------现金违规v1.0---------------------------------

                # 当前帧所有'staff'、'cash'预测框坐标
                cash_boxes, staff_boxes = get_boxes(det) #list: [[x1,y1,x2,y2],...]
                log.debug("==> cash_boxes:  %s", cash_boxes)
                log.debug("==> staff_boxes: %s", staff_boxes)

                # 每个围栏判断是否违规
                for i, fence in enumerate(fences): 
                    log.debug('==> fence_{}: ------'.format(i))
                    
                    # 当前围栏区域下预测框，与所有预测框求交集
                    cash_boxes_in = box_inter(fence, cash_boxes)
                    staff_boxes_in = box_inter(fence, staff_boxes)
                    
                    # 判断当前围栏中是否存在现金违规
                    if cash_illegal(cash_boxes_in, staff_boxes_in):
                        log.debug('==> fence_{}: abnormal，{} cash, {} staff'.format(i,cash_boxes_in,staff_boxes_in))
                        puter(fence_result[i], 1)
                    else:
                        log.debug('==> fence_{}: normal，{} cash, {} staff'.format(i,cash_boxes_in,staff_boxes_in))
                        puter(fence_result[i], 0)
                    
                    # 历史多帧统计决定是否最终违规
                    # 规则：若连续N帧中违规帧率大于阈值，则违规
                    if sum(fence_result[i]) / max_save_frame > abnormal_thres:
                        log.info('==> CASH WARN! fence:{} sum:{}'.format(i, sum(fence_result[i])))
                
                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                view_img = True
                if view_img:
                    im0 = show_image(det, im0, names, colors, fence_color, None, frame)
                    
                    # 实时展示，会降低FPS
                    if False:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(0)

                # Save results (image with detections)
                save_img = False
                if save_img or view_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)

                ## --------------------------------------------------------------------------------

                ## Print results
                #for c in det[:, -1].unique():
                #    n = (det[:, -1] == c).sum()  # detections per class
                #    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results
                #for *xyxy, conf, cls in reversed(det):
                #    if save_txt:  # Write to file
                #        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                #        with open(txt_path + '.txt', 'a') as f:
                #            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                #    if save_img or view_img:  # Add bbox to image
                #        label = f'{names[int(cls)]} {conf:.2f}'
                #        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    # 获取当前摄像头电子围栏信息
    fences = get_fence(cam_id='0001', scene='cash')
    log.info('=>fences: {}'.format(fences))

    # 电子围栏的违规统计变量
    fence_result = {}
    for i in range(len(fences)):
        fence_result[i] = []
        
    # 规则判断超参，可根据实际情况调整
    max_save_frame = 60  #存储多少帧结果
    abnormal_thres = 0.5 #违规帧率

    # 运行推理
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
