import cv2
import numpy as np

RED_LOW1= (0, 100, 80)
RED_HIGH1 = (10, 255, 255)
RED_LOW2 = (170, 100, 80)
RED_HIGH2 = (180, 255, 255)

#4.2 (segment red markers)
def apply_red_mask_hsv(hsv):
    m1 = cv2.inRange(hsv, np.array(RED_LOW1,  np.uint8), np.array(RED_HIGH1, np.uint8))
    m2 = cv2.inRange(hsv, np.array(RED_LOW2,  np.uint8), np.array(RED_HIGH2, np.uint8))
    mask = cv2.bitwise_or(m1, m2)
    structe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, structe, iterations=1)
    mask = cv2.dilate(mask, structe, iterations=1)
    return mask

#4.3 (marionette)
def find_red_centroids(mask):
    conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in conts:
        if cv2.contourArea(c) < 20:
            continue
        M = cv2.moments(c)
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centers.append((cx, cy))
    return centers

def assign_joints(centers, prev=None):
    joints = {'torso': None, 'lh': None, 'rh': None, 'lf': None, 'rf': None}
    if not centers:
        return joints, prev
    pts = np.array(centers, dtype=np.int32)
    torso = tuple(np.mean(pts, axis=0).astype(int).tolist())
    # print(torso)
    upper = pts[pts[:,1] < torso[1]]
    lower = pts[pts[:,1] >= torso[1]]
    # hands
    if len(upper) >= 2:
        lh = tuple(upper[np.argmin(upper[:,0])].tolist())
        rh = tuple(upper[np.argmax(upper[:,0])].tolist())
    elif len(upper) == 1:
        only = tuple(upper[0].tolist())
        lh, rh = (only, None) if only[0] < torso[0] else (None, only)
    else:
        lh = rh = None
    # feet
    if len(lower) >= 2:
        lf = tuple(lower[np.argmin(lower[:,0])].tolist())
        rf = tuple(lower[np.argmax(lower[:,0])].tolist())
    elif len(lower) == 1:
        only = tuple(lower[0].tolist())
        lf, rf = (only, None) if only[0] < torso[0] else (None, only)
    else:
        lf = rf = None
    joints.update({'torso': torso, 'lh': lh, 'rh': rh, 'lf': lf, 'rf': rf})
    if prev is not None:
        for k in joints:
            if joints[k] is not None and prev.get(k) is not None:
                p = np.array(prev[k], dtype=np.float32)
                c = np.array(joints[k], dtype=np.float32)
                joints[k] = tuple((0.5*p + 0.5*c).astype(int).tolist())
    return joints, joints

def draw_marionette(img, joints):
    body_col, limb_col = (200,200,200), (100,200,255)
    shorts_col = (189, 96, 21)
    t = joints['torso']
    lh = joints['lh']
    rh = joints['rh']
    lf = joints['lf']
    rf = joints['rf']
    if t is not None and lh is not None:
        cv2.line(img, t, lh, limb_col, 8)
        mid = ((t[0] + lh[0]) // 2, (t[1] + lh[1]) // 2)
        cv2.line(img, t, mid, body_col, 8)
    if t is not None and rh is not None:
        cv2.line(img, t, rh, limb_col, 8)
        mid2 = ((t[0] + rh[0]) // 2, (t[1] + rh[1]) // 2)
        cv2.line(img, t, mid2, body_col, 8)
    if t is not None and lf is not None:
        cv2.line(img, t, lf, limb_col, 10)
        mid3 = ((t[0] + lf[0]) // 2, (t[1] + lf[1]) // 2)
        cv2.line(img, t, mid3, shorts_col, 10)
    if t is not None and rf is not None:
        cv2.line(img, t, rf, limb_col, 10)
        mid4 = ((t[0] + rf[0]) // 2, (t[1] + rf[1]) // 2)
        cv2.line(img, t, mid4, shorts_col, 10)
    if t is not None:
        tx, ty = t; cv2.rectangle(img, (tx-18, ty-28), (tx+18, ty+28), body_col, -1)
        head_cx = tx
        head_cy = ty - 28 - 4 - 14
        cv2.circle(img, (head_cx, head_cy), 15, limb_col, -1)

#4.4 Intelligent objects
class Wanderer:
    def __init__(self, W, H, color=(255,200,50)):
        self.W, self.H = W, H
        rng = np.random.default_rng()
        self.pos = np.array([rng.uniform(0.2*W, 0.8*W), rng.uniform(0.2*H, 0.8*H)], np.float32)
        self.vel = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3)], np.float32)
        self.color = color
        self.radius = 14

    def step(self):
        self.vel += np.random.uniform(-1, 1, 2)
        s = np.linalg.norm(self.vel) + 0.000001
        if s > 5:
            self.vel *= 5/s
        self.pos += self.vel
        
        if self.pos[0] < self.radius or self.pos[0] > self.W-self.radius:
            self.vel[0] *= -1
            self.pos[0] = np.clip(self.pos[0], self.radius, self.W-self.radius)
        if self.pos[1] < self.radius or self.pos[1] > self.H-self.radius:
            self.vel[1] *= -1
            self.pos[1] = np.clip(self.pos[1], self.radius, self.H-self.radius)

    def draw(self, im):

        x, y = int(self.pos[0]), int(self.pos[1])

        medkit_sprite = cv2.imread('assets/Medkit_1.webp', cv2.IMREAD_UNCHANGED)
        kH = 40
        resized_medkit = cv2.resize(medkit_sprite, (40, 40))

        spr_bgr = resized_medkit[..., :3]
        spr_mask = resized_medkit[..., 3]
        roi = im[y:y+kH, x:x+kH]
        cv2.copyTo(spr_bgr, spr_mask, roi)

    def get_pos(self):
        return tuple(self.pos.tolist())

class Seeker:
    def __init__(self, W, H, color=(60,180,255)):
        self.W, self.H = W, H
        rng = np.random.default_rng()
        self.pos = np.array([rng.uniform(0.2*W, 0.8*W), rng.uniform(0.2*H, 0.8*H)], np.float32)
        self.vel = np.zeros(2, np.float32)
        self.max_speed = 6.0
        self.k = 0.6
        self.color = color
        self.size = 24
        self.avoid_frames = 0

    def avoid(self, n=20):
        self.avoid_frames = n

    def step(self, target_xy):
        target = np.array(target_xy, np.float32)
        dirv = target - self.pos
        d = np.linalg.norm(dirv) + 0.000001
        u = dirv / d
        g = -abs(self.k) if self.avoid_frames > 0 else abs(self.k)
        self.vel += g * u
        s = np.linalg.norm(self.vel) + 0.000001
        if s > self.max_speed:
            self.vel *= self.max_speed / s
        self.pos += self.vel
        self.pos[0] = np.clip(self.pos[0], 0, self.W-1)
        self.pos[1] = np.clip(self.pos[1], 0, self.H-1)
        if self.avoid_frames > 0:
            self.avoid_frames -= 1

    def draw(self, im):
        x, y = int(self.pos[0]), int(self.pos[1])

        knife_sprite = cv2.imread('assets/knife4.png')
        kH = 40
        resized_knife = cv2.resize(knife_sprite, (40, 40))

        mask_bg = cv2.inRange(resized_knife, (240,240,240), (255,255,255))
        mask = cv2.bitwise_not(mask_bg)
        roi = im[y:y+kH, x:x+kH]
        cv2.copyTo(resized_knife, mask, roi)

    def get_pos(self):
        return tuple(self.pos.tolist())
    
def collided(p1, r1, p2, r2):
    p1, p2 = np.array(p1), np.array(p2)
    return np.linalg.norm(p1-p2) <= (r1 + r2)

def main():
    cap = cv2.VideoCapture("assets/Opt1-MarionetteMovements.mov")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("HEIGHT: " ,H,"WIDTH: ", W)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter("out.mp4", fourcc, 30, (W, H))

    prev_joints = None
    frame_idx = 0
    total_frame = 30 * 60

    wanderer = Wanderer(W, H)
    seeker   = Seeker(W, H)
    flash_frames = 0

    HP = 100
    damned = False

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    while frame_idx < total_frame:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        # img = cv2.imread('assets/mansion_front.png')
        img = cv2.imread('assets/mansion_bedroom.png')
        resized_img = cv2.resize(img, (W, H))
        if frame_idx > 200:
            img = cv2.imread('assets/mansion_dining.png')
            resized_img = cv2.resize(img, (W, H))
        if frame_idx > 600:
            img = cv2.imread('assets/mansion_main_hall.png')
            resized_img = cv2.resize(img, (W, H))
        if frame_idx > 1200:
            img = cv2.imread('assets/mansion_front.png')
            resized_img = cv2.resize(img, (W, H))
        if frame_idx > 1650:
            img = cv2.imread('assets/city.jpg')
            resized_img = cv2.resize(img, (W, H))
        if HP <= 0:
            if frame_idx < 1200:
                damned = True
            if damned == False:
                img = cv2.imread('assets/heaven.jpg')
                resized_img = cv2.resize(img, (W, H))
            else:
                img = cv2.imread('assets/hell.webp')
                resized_img = cv2.resize(img, (W, H))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = apply_red_mask_hsv(hsv)
        centers = find_red_centroids(mask)
        joints, prev_joints = assign_joints(centers, prev=prev_joints)
        draw_marionette(mask, joints)
        draw_marionette(resized_img, joints)

        #intelligent objs
        if frame_idx > 300 and HP > 0 and frame_idx <= 1650:
            wanderer.step()
            wanderer.draw(resized_img)
            torso = joints['torso']
            seeker.step(torso)
            seeker.draw(resized_img)

            if joints['torso'] is not None:
                if collided(seeker.get_pos(), 24, joints['torso'], 18):
                    flash_frames = 8
                    HP -= 1
                    seeker.avoid(20)
                if collided(wanderer.get_pos(), 14, joints['torso'], 18):
                    HP += 2
                    if HP > 100:
                        HP = 100
            if flash_frames > 0:
                # cv2.circle(resized_img, joints['torso'], 28, (0,255,255), 3)
                
                x, y = joints['torso'][0], joints['torso'][1]

                blood_sprite = cv2.imread('assets/blood.jpg')
                kH = 60
                resized_blood = cv2.resize(blood_sprite, (60, 60))

                mask_bg = cv2.inRange(resized_blood, (220,220,220), (255,255,255))
                mask = cv2.bitwise_not(mask_bg)
                roi = resized_img[y-25:y+kH-25, x-25:x+kH-25]
                cv2.copyTo(resized_blood, mask, roi)

                resized_img[:] = cv2.add(resized_img, np.full_like(resized_img, 25))
                flash_frames -= 1

        cv2.putText(resized_img, "Health: " + str(HP), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 1)
        cv2.putText(resized_img, "SID530657486_Asgmt2Opt1", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        bgrmask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        writer.write(resized_img)
        print(frame_idx)
        frame_idx +=1

if __name__ == '__main__':
    main()