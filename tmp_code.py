
import marshal
def torso_visible(scores):
    return ((scores[11] > 0.2 or
            scores[12] > 0.2) and
            (scores[5] > 0.2 or
            scores[6] > 0.2))
def determine_torso_and_body_range(x, y, scores, center_x, center_y):
    torso_joints = [5, 6, 11, 12]
    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for i in torso_joints:
        dist_y = abs(center_y - y[i])
        dist_x = abs(center_x - x[i])
        if dist_y > max_torso_yrange:
            max_torso_yrange = dist_y
        if dist_x > max_torso_xrange:
            max_torso_xrange = dist_x
    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for i in range(17):
        if scores[i] < 0.2:
            continue
        dist_y = abs(center_y - y[i])
        dist_x = abs(center_x - x[i])
        if dist_y > max_body_yrange:
            max_body_yrange = dist_y
        if dist_x > max_body_xrange:
            max_body_xrange = dist_x
    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]
def determine_crop_region(scores, x, y):
    if torso_visible(scores):
        center_x = (x[11] + x[12]) // 2
        center_y = (y[11] + y[12]) // 2
        max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange = determine_torso_and_body_range(x, y, scores, center_x, center_y)
        crop_length_half = max(max_torso_xrange * 1.9, max_torso_yrange * 1.9, max_body_yrange * 1.2, max_body_xrange * 1.2)
        crop_length_half = int(round(min(crop_length_half, max(center_x, 1152 - center_x, center_y, 648 - center_y))))
        crop_corner = [center_x - crop_length_half, center_y - crop_length_half]
        if crop_length_half > max(1152, 648) / 2:
            return {'xmin': 0, 'ymin': -252, 'xmax': 1152, 'ymax': 900, 'size': 1152}
        else:
            crop_length = crop_length_half * 2
            return {'xmin': crop_corner[0], 'ymin': crop_corner[1], 'xmax': crop_corner[0]+crop_length, 'ymax': crop_corner[1]+crop_length, 'size': crop_length}
    else:
        return {'xmin': 0, 'ymin': -252, 'xmax': 1152, 'ymax': 900, 'size': 1152}
def pd_postprocess(inference, crop_region):
    size = crop_region['size']
    xmin = crop_region['xmin']
    ymin = crop_region['ymin']
    xnorm = []
    ynorm = []
    scores = []
    x = []
    y = []
    for i in range(17):
        xn = inference[3*i+1]
        yn = inference[3*i]
        xnorm.append(xn)
        ynorm.append(yn)
        scores.append(inference[3*i+2])
        x.append(int(xmin + xn * size)) 
        y.append(int(ymin + yn * size)) 
    next_crop_region = determine_crop_region(scores, x, y) if True else init_crop_region
    return x, y, xnorm, ynorm, scores, next_crop_region
node.warn("Processing node started")
init_crop_region = {'xmin': 0, 'ymin': -252, 'xmax': 1152, 'ymax': 900, 'size': 1152}
crop_region = init_crop_region
result_buffer = Buffer(759)
while True:
    cfg = ImageManipConfig()
    points = [
        [crop_region['xmin'], crop_region['ymin']],
        [crop_region['xmax']-1, crop_region['ymin']],
        [crop_region['xmax']-1, crop_region['ymax']-1],
        [crop_region['xmin'], crop_region['ymax']-1]]
    point2fList = []
    for p in points:
        pt = Point2f()
        pt.x, pt.y = p[0], p[1]
        point2fList.append(pt)
    cfg.setWarpTransformFourPoints(point2fList, False)
    cfg.setResize(256, 256)
    cfg.setFrameType(ImgFrame.Type.RGB888p)
    node.io['to_manip_cfg'].send(cfg)
    inference = node.io['from_pd_nn'].get().getLayerFp16("Identity")
    x, y, xnorm, ynorm, scores, next_crop_region = pd_postprocess(inference, crop_region)
    result = {"x":x, "y":y, "xnorm":xnorm, "ynorm":ynorm, "scores":scores, "next_crop_region":next_crop_region}
    result_serial = marshal.dumps(result)
    result_buffer.getData()[:] = result_serial
    node.io['to_host'].send(result_buffer)
    crop_region = next_crop_region
