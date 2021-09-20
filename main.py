import pygame, thorpy
from pygame.math import Vector2 as V2
import random
import math

W,H = 1200,700
FPS = 80
DT = 0.08
GRAVITY = 9.81

def draw_vect(surface, p0, color, v):
    pygame.draw.line(surface, color, p0, p0 + v, 3)

def sgn(x):
    if x > 0:
        return 1.
    return -1.

def rand_sgn():
    if random.random() < 0.5:
        return 1
    return -1

def angle_diff(angle1, angle2):
    diff = abs(angle1-angle2)
    if diff > 180:
        return 360 - diff #this is an absolute value
    return diff

def to_positive_deg(angle):
    if angle < 0:
        return (360+angle)%360.
    else:
        return angle%360.

def to_small_form(angle):
    if angle > 0:
        if angle > 180:
            return angle - 360.
        return angle
    else:
        if angle < -180:
            return 360 + angle
        return angle

class Plane:

    def __init__(self):
        #aerodynamics
        self.scx = 1.
        self.scz = 1.
        self.scx_90 = 3.
        self.k = 0.1 #railing
        #engine and commands
        self.fuel = 1.
        self.power = 10000.
        self.consumption = 1e-2
        self.command_torque = 3.
        self.throttling = False
        self.damage = 0
        self.max_damage = 1000
        #physics translation
        self.mass = 1000.
        self.pos = V2(0,H//2)
        self.vel = V2(1., 0)
        self.force = V2()
        self.vel_stall = 30.
        #physics rotation
        self.I = 1.
        self.orientation = V2()
        self.aoa = 0. #angle of attack
        self.omega = 0.
        self.torque = 0.
        self.rot_friction_coeff = 0.01
        #geometry
        self.L = 80
        self.H = self.L//4
        self.D = max(self.L, self.H)
##        self.img = pygame.Surface((self.D,)*2)
##        self.img.fill((255,255,255))
##        r = pygame.Rect(0, 0, self.L, self.H)
##        r.center = self.img.get_rect().center
##        pygame.draw.rect(self.img, (255,255,0), r)
##        r.x += 3*self.L//4
##        r.y -= self.H//2
##        pygame.draw.rect(self.img, (0,0,255), r)
        self.img = planes[3]
        self.img = pygame.transform.scale(self.img, ((self.D,)*2))
        self.img = self.img.convert_alpha()
        self.img_orig = self.img.copy()

    def iterate_points(self):
        oL = self.orientation*self.L*0.5
        yield self.pos + oL
        yield self.pos - oL
        oH = self.orientation.rotate(90.)*self.H*0.5
        yield self.pos + oH
        yield self.pos - oH
        yield self.pos

    def get_drag_coeff(self):
        a = abs(self.aoa)
        if a > 90:
            a = 90
##        ratio = a/90.
##        return (1.-ratio) * self.scx + ratio * self.scx_90
        #could be a sine more realistically
        a_rad = a * math.pi / 180.
        return self.scx + self.scx_90*math.sin(a_rad)

    def get_command_torque(self):
        v = self.vel.length()
        if v > self.vel_stall:
            return self.command_torque
        return self.command_torque * ( 1. - v/self.vel_stall)

    def get_altitude(self):
        return H//2-self.pos.y

    def update_physics(self):
        #update angle of attack
        self.aoa = to_small_form(self.vel.angle_to(self.orientation))
##        print(self.aoa)
##        print(self.vel.length())
##        print(self.get_drag_coeff())
##        print(self.get_altitude())
        #compute drag ##########################################################
        v_sqr = self.vel.length()**2
        drag = -self.get_drag_coeff() * v_sqr
        v2_drag = self.vel.normalize()*drag
        #compute lift ##########################################################
        lift = -self.scz * v_sqr * (1. - abs(self.aoa)/180.)
        v2_lift = self.orientation.rotate(90.)*lift
        #align vel to orientation
        if self.vel.length() > self.vel_stall:
            delta_angle_vel = self.k * self.aoa
            self.vel.rotate_ip(delta_angle_vel)
            self.orientation.rotate_ip(-3.*delta_angle_vel)
        #add forces ############################################################
        self.force += v2_drag + v2_lift + V2(0,self.mass*GRAVITY)
        self.vel += DT*self.force / self.mass
        delta = DT*self.vel
        self.pos += delta
        self.force = V2()
        #add torque ############################################################
        self.omega += DT * self.torque / self.I
        self.omega -= self.omega * self.rot_friction_coeff
        self.orientation.rotate_ip(DT * self.omega)
        self.torque = 0.
        return delta

    def update_command(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            self.torque += self.get_command_torque()
        if keys[pygame.K_LEFT]:
            self.torque -= self.get_command_torque()
        if keys[pygame.K_SPACE]:
            self.force += self.orientation * self.power
            self.throttling = True
            self.fuel -= self.consumption
            smokegen2.generate(V2(player_center)-player.orientation*player.L/2)
        else:
            self.throttling = False

    def align_img_to_orientation(self):
        self.img = pygame.transform.rotate(self.img_orig,
                                            self.orientation.angle_to(V2(1,0)))

    def is_dead(self):
        return self.get_altitude() < 0 or self.damage > self.max_damage

iteration = 0

def make_debris():
    angle = 0
    spread = 180 #spread of debris directions
    vel = int(player.vel.length()*DT)
    debrisgen.generate( player_center, #position
                        n=1, #number of debris
                        v_range = (vel, 4*vel),
##                        v_range=(10,50), #translational velocity range
                        omega_range=(5,25), #rotational velocity range
                        angle_range=(angle-spread,angle+spread))

def make_debris_explosion():
##    angle = 120 #go to left
    angle = 180
    spread = 30 #spread of debris directions
    vel = 40
    debrisgen2.generate( player_center, #position
                        n=40, #number of debris
                        v_range = (vel, 4*vel),
##                        v_range=(10,50), #translational velocity range
                        omega_range=(5,25), #rotational velocity range
                        angle_range=(angle-spread,angle+spread))

def make_coin_explosion():
##    angle = 120 #go to left
    angle = 180
    spread = 180 #spread of debris directions
    vel = 20
    debrisgen3.generate( player_center, #position
                        n=100, #number of debris
                        v_range = (vel, 4*vel),
##                        v_range=(10,50), #translational velocity range
                        omega_range=(5,25), #rotational velocity range
                        angle_range=(angle-spread,angle+spread))


mon = thorpy.Monitor()
def draw():
    mon.append("b")
    global done, holes, player_won
    player_velnorm = player.vel.length()
    player_velDT = player.vel*DT
    mon.append("c")
    player.align_img_to_orientation()
    player_rect = player.img.get_rect()
    player_rect.center = player_center
    #Draw sky ##################################################################
    SPACE_ALTITUDE = 5000
    f = 1. - player.get_altitude() / SPACE_ALTITUDE
    if f > 1:
        f = 1
    elif f < 0:
        f = 0
    screen.fill((150*f,150*f,255*f))
    star.set_alpha((1-f)*255)
    screen.blits(stars)
    moon.set_alpha((1-f)*250 + 5)
    screen.blit(moon, (100,80))
    mon.append("d")
    #Draw far field ############################################################
##    screen.blits(far_field2)
    for img, pos in far_field2:
        screen.blit(img, pos)
        pos[1] -= player_velDT.y/ 100.
    for pos,img in far_field:
        screen.blit(img, pos)
        pos[0] -= player_velDT.x / 30.
        pos[1] -= player_velDT.y / 30.
    mon.append("e")
    #Draw and detect obstacles #################################################
    idx_obstacle = int(player.pos.x // LUT_DIVIDOR)
    m_idx = idx_obstacle-1
    if m_idx < 0:
        m_idx = 0
    M_idx = idx_obstacle+3
    if M_idx > len(obstacles_LUT):
        M_idx = len(obstacles_LUT)
    for idx in range(m_idx, M_idx):
        for x,y,img in obstacles_LUT[idx]:
            o_rect = img.get_rect()
            o_rect.topleft = x-player.pos.x, y-player.pos.y
            screen.blit(img, o_rect)
            if abs(idx - idx_obstacle) < 2:
                o_rect.topleft = x-player_center[0], y-player_center[1]
                for point in player.iterate_points():
                    collision = False
                    if o_rect.collidepoint(point):
                        mod = int(400. / (player_velnorm+0.1))
                        if iteration%mod == 0:
                            s = pygame.Surface(hole_size)
                            s.fill((150*f,150*f,255*f))
        ##                    holes.append((player.pos.x+player_center[0], player.pos.y+player_center[1], s))
                            left = player_center[0] - x + player.pos.x - hole_size[0]//2
                            top = player_center[1] - y + player.pos.y - hole_size[1]//2
                            img.blit(s, (left, top))
                        if random.random() < 0.5:
                            make_debris()
                        player.vel *= 0.995
                        player.damage += 1
                        collision = True
                        break
                    if collision:
                        break
    mon.append("f")
    #Draw and detect coins #####################################################
    idx_coin = int(player.pos.x // LUT_DIVIDOR)
    m_idx = idx_coin-1
    if m_idx < 0:
        m_idx = 0
    M_idx = idx_coin+3
    if M_idx > len(coins_LUT):
        M_idx = len(coins_LUT)
    for idx in range(m_idx, M_idx):
        to_remove = -1
        for i, data in enumerate(coins_LUT[idx]):
            x,y,rect = data
            rect.topleft = x-player.pos.x, y-player.pos.y
            screen.blit(coin_img, rect)
            if abs(idx - idx_coin) < 2:
                rect.topleft = x-player_center[0], y-player_center[1]
                for point in player.iterate_points():
                    if rect.collidepoint(point):
                        to_remove = i
                        break
            if to_remove > -1:
                break
        if to_remove > -1:
            coins_LUT[idx].pop(to_remove) #TODO: voir perf si mieux de juste ignorer les coins mangés ?
            break
    mon.append("g")
    #Draw jerricans ############################################################
    rect_jerrican = jerrican_img.get_rect()
    to_remove = -1
    for i, pos in enumerate(jerricans):
        x,y = pos
        x_img = x - player.pos.x
        y_img = y - player.pos.y
        screen.blit(jerrican_img, (x_img,y_img))
        rect_jerrican.topleft = x - player_center[0], y - player_center[1]
        for point in player.iterate_points():
            if rect_jerrican.collidepoint(point):
                to_remove = i
                break
        if to_remove > -1:
            break
    if to_remove > -1:
        jerricans.pop(to_remove)
    mon.append("h")
    #Draw finish line ##########################################################
    x = finish_coord - player.pos.x
    pygame.draw.rect(screen, (0,0,0), pygame.Rect(x, 0, x+100, H))
    if x < player_rect.right:
        player_won = True
        if finished < FPS//8:
            make_coin_explosion()
    #Draw debris and smokes ####################################################
    if finished < 0:
        screen.blit(player.img, player_rect)
    elif player_won:
        debrisgen3.kill_translate_update_draw(screen_rect, V2(0,4), DT, screen)
    else:
        debrisgen2.kill_translate_update_draw(screen_rect, V2(0,4), DT, screen)
    debrisgen.kill_translate_update_draw(screen_rect, -player_velDT + V2(0,2),
                                            DT, screen)
    smokegen2.kill_update_draw(-V2(player_velDT), screen)
    mon.append("i")
    #Draw clouds ###############################################################
    for pos,dx,img in clouds:
        delta = pos - player.pos
        screen.blit(img, (delta.x,delta.y))
        pos.x += dx
        if delta.x < -300:
            pos.x += 1.5*W
            pos.y += rand_sgn()*100
        elif delta.x > W+300:
            pos.x -= 1.5*W
            pos.y += rand_sgn()*100
    mon.append("j")
    #Draw ground ###############################################################
    y_ground = H - player.pos.y
    pygame.draw.rect(screen, (0,150,0), pygame.Rect(0,y_ground,W,H))
    #Draw wind #################################################################
    if player_velnorm > 40:
        wind_img = pygame.transform.rotate(wind_img_orig, player.orientation.angle_to(V2(1,0)))
        wind_img2 = pygame.transform.rotate(wind_img_orig2, player.orientation.angle_to(V2(1,0)))
        wind_imgs = [wind_img, wind_img2]
        n = int(player_velnorm//8)
        if n > 20:
            n = 20
        elif n < 1:
            n = 1
        for i,wind_pos in enumerate(winds[0:n]):
            wind_pos -= 2*player_velDT
            screen.blit(wind_imgs[i%2], wind_pos)
            wind_pos.x %= W + random.randint(5,20)
            wind_pos.y %= H + random.randint(5,20)
    #Draw altitude alert #######################################################
    if H//2 < player.get_altitude() < 1.5*H:
        if player.vel.y > 20.:
            f = 0.5*math.sin(iteration*0.1) + 0.5
            ground_alert.set_alpha(f*255)
            screen.blit(ground_alert, ((W-ground_alert.get_width())//2, H-60))
    mon.append("k")



winds = [V2(random.randint(0,W), random.randint(0,H)) for i in range(20)]

ground_alert = pygame.Surface((3*W//4, 40))
ground_alert.fill((255,0,0))

random.seed(0)

star = pygame.Surface((3,3))
star.fill((255,255,0))
star.set_alpha(0)


def get_player_center():
    x0, y0 = 300, H//2
    f = 2.
    return x0 + player.vel[0]/f, y0 + player.vel[1]/f


DMAX = 10000
N_OBSTACLES = 20
obstacles = []
obstacles_imgs = {}
COLOR_OBSTACLE = (100,)*3


OBS_W = 100, 200
OBS_H = 100, 200
def add_obstacle(x,y,f):
    w = int(random.randint(OBS_SIZE[0],OBS_SIZE[1]) * f)
    h = int(random.randint(H//4, H//2) * f)
    #
    r = pygame.Rect(0,0,w,h)
    img = pygame.Surface((w,h))
    img.fill(COLOR_OBSTACLE)
    pygame.draw.rect(img, (0,0,0), r, 3)
    r.topleft = x,y
    okay = True
    for x2,y2,img2 in obstacles:
        r2 = pygame.Rect((x2,y2), img2.get_size())
        if r2.colliderect(r):
            okay = False
            break
    if okay:
        obstacles.append((x,y,img))

def altitude_to_y(a):
##    H/2 - y = a
    return H//2 - a

##profil = [(500, -1000+H), (500+2*W, -10+H)]
OBSTACLE_DENSITY = 1e-3
##dans build profile, representer sous le curseur la taille des blocs de ce niveau de densite



##jerrican ajoutes manuellement
##coin ajoutes automatique.

COIN_RAND_X = -200, 300
COIN_RAND_Y = -200, 300
coins = []
def add_profil(profil, size_factor, add_coins=False):
    for i in range(len(profil)-1):
        x0,y0,density = profil[i]
        x1,y1,osef = profil[i+1]
        n = int((x1-x0)*OBSTACLE_DENSITY*density)
        for k in range(n):
            x = x0 + k/n * (x1-x0)
            y = y0 + k/n * (y1-y0)
            add_obstacle(x,altitude_to_y(y)+H//2,size_factor)
            if add_coins:
                for i in range(2):
                    dx = random.randint(COIN_RAND_X[0], COIN_RAND_X[1])
                    dy = random.randint(COIN_RAND_Y[0], COIN_RAND_Y[1])
                    coins.append((x+dx,altitude_to_y(y)+H//2-50+dy))


WORLD_SIZE_FACTOR_X = 4
WORLD_SIZE_FACTOR_Y = WORLD_SIZE_FACTOR_X
def build_profils():
    screen.fill((0,)*3)
    profil_finished = False
    profils = [[],[],[],[],[]]
    i_profil = 0
    colors = [(255,0,0), (0,255,0), (0,0,255), (100,100,100),  (255,255,255)]
    density = [1.,        2.,          4.,      4.,    0.5]
    pygame.draw.rect(screen, colors[i_profil],
                                    pygame.Rect(W//2,0,10,10))
    hh = H//WORLD_SIZE_FACTOR_Y
    for i in range(WORLD_SIZE_FACTOR_Y+1):
        pygame.draw.line(screen, (50,)*3, (0,H-i*hh), (W,H-i*hh))
    pygame.display.flip()
    while not profil_finished:
        for e in pygame.event.get():
            if e.type == pygame.MOUSEBUTTONDOWN:
                x,y = pygame.mouse.get_pos()
                r = pygame.Rect(0,0,5,5)
                r.center = x,y
                x *= WORLD_SIZE_FACTOR_X
                y = (H-y)*WORLD_SIZE_FACTOR_Y
                print(x,y)
                pygame.draw.rect(screen, colors[i_profil], r)
                pygame.display.flip()
                print(i_profil)
                profils[i_profil].append((x, y, density[i_profil]))
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_s:
                    return profils
                elif e.key == pygame.K_SPACE:
                    i_profil += 1
                    i_profil %= len(colors)
                    print("Density:",density[i_profil])
                    pygame.draw.rect(screen, colors[i_profil],
                                    pygame.Rect(W//2,0,10,10))
                    pygame.display.flip()


app = thorpy.Application((W, H), "Power glider")
screen = app.get_screen()
screen_rect = screen.get_rect()


name = "race1"
LOAD = True
import pickle
if LOAD:
    profils = pickle.load(open(name + ".dat", "rb" ))
else:
    profils = build_profils()
    pickle.dump(profils, open(name + ".dat", "wb" ))

for i,profil in enumerate(profils[0:-1]):
    add_profil(profil, add_coins=i==0, size_factor=(i+0.5))
jerricans = [(x,y) for x,y,d in profils[-1]]

jerrican_img = pygame.transform.scale(pygame.image.load("jerrican.png"), (40,40))
jerrican_img.set_colorkey((255,)*3)
coin_img = pygame.transform.scale(pygame.image.load("coin.png"), (40,40)).convert()
coin_img.set_colorkey((255,)*3)

clouds = []
for i in range(10):
    x = random.randint(0,DMAX)
    y = random.randint(-2*H, 0)
    w = random.randint(50, 300)
    h = random.randint(50, w)
    img = pygame.Surface((w,h))
    c = random.choice([220,250])
    img.fill((c,)*3)
    img.set_alpha(127)
    dx = random.choice([-2,-1])
    clouds.append((V2(x,y), dx, img))

stars = []
for i in range(30):
    x = random.randint(0,W)
    y = random.randint(0,2*H//3)
    stars.append((star, (x,y)))

obstacles.sort(key=lambda x:x[0])
finish_coord = obstacles[-1][0]
##obstacles[-1][-1].fill((255,0,0))

moon = pygame.transform.scale(pygame.image.load("moon.png"), (16, 16)).convert()
moon.set_colorkey((0,0,0))
moon = pygame.transform.scale(moon, (100,100))


smokegen2 = thorpy.fx.get_fire_smokegen(n=50, color=(200,255,155), grow=0.4, size0=(10,10))
debrisgen = thorpy.fx.get_debris_generator(duration=200, #nb. frames before die
                                    color=COLOR_OBSTACLE,
                                    max_size=20)
debrisgen2 = thorpy.fx.get_debris_generator(duration=1000, #nb. frames before die
                                    color=(255,255,0),
                                    max_size=20)
debrisgen3 = thorpy.fx.get_debris_generator(duration=1000, #nb. frames before die
                                    color=(255,255,0),
                                    max_size=20,
                                    samples=[coin_img, coin_img])

PLAYER_COLOR = (0,0,255)
all_planes = pygame.image.load("planes.png")
thorpy.change_color_on_img_ip(all_planes, (255,255,0), PLAYER_COLOR)
thorpy.change_color_on_img_ip(all_planes, (224,224,0), tuple([0.6*c for c in PLAYER_COLOR]))
planes = []
for i in range(all_planes.get_width()//32):
    s = pygame.Surface((32,32))
    s.fill((255,)*3)
    s.blit(all_planes, (-i*32,0))
    s.set_colorkey((255,)*3)
    planes.append(s)


player = Plane()
player.pos.y = altitude_to_y(profils[0][0][1] + 500)
player.vel.x = 50
player.scx = 0.03
player.scx_90 = 2.
player.scz = 0.1
player.I = 0.1
player.orientation = player.vel.normalize()


LUT_DIVIDOR = W
obstacles_LUT = [[] for i in range(W*WORLD_SIZE_FACTOR_X//LUT_DIVIDOR)]
for x,y,img in obstacles:
    idx = int(x // LUT_DIVIDOR)
    obstacles_LUT[idx].append((x,y,img))
coins_LUT = [[] for i in range(W*WORLD_SIZE_FACTOR_X//LUT_DIVIDOR)]
for x,y in coins:
    idx = int(x // LUT_DIVIDOR)
    r = coin_img.get_rect()
    coins_LUT[idx].append((x,y,r))



far_field = []
far_field2 = []
def add_far_field(x,w,h,c,k):
    if k != 0:
        h += 100
    s = pygame.Surface((w,h+200))
    s.fill((c,)*3)
    if k == 0:
        far_field.append(([x,H-h],s))
    else:
        far_field2.append((s,[x,H-h+100]))
for k in range(2):
    if k == 0:
        wmax = W
        c = 50
        c2 = 25
    else:
        wmax = 2*W
        c = 80
        c2 = 60
    for i in range(15):
        x = random.randint(0,wmax)
        w = random.randint(50,150)
        h = random.randint(50,200)
        add_far_field(x,w,h,c,k)
        if random.random() < 0.2: #add twin
            w2 = random.randint(50,150)
            h2 = random.randint(h//3,h//2)
            add_far_field(x+random.randint(w//2,3*w//4), w2, h2,c2,k)
            if random.random() < 0.4: #add antenna
                add_far_field(x+random.randint(0,w2-10),10,
                                random.randint(int(1.2*h2),int(1.5*h2)),c2,k)
        if random.random() < 0.4: #add antenna
            add_far_field(x+random.randint(0,w-10),10,
                            random.randint(int(1.2*h),int(1.5*h)),c,k)

wind_img_orig = pygame.Surface((20,20))
pygame.draw.rect(wind_img_orig, (255,255,255), pygame.Rect(0,9,20,3))
wind_img_orig.set_colorkey((0,0,0))
wind_img_orig2 = pygame.transform.scale(wind_img_orig, (10,10))

##wind_img_orig = pygame.image.load("wind.png").convert()
##wind_img_orig.set_colorkey((0,0,0))
##wind_img_orig = pygame.transform.scale(wind_img_orig, (100,100))
##wind_img_orig.set_alpha(100)
##wind_img_orig2 = pygame.transform.flip(wind_img_orig, False, True)

player_center = get_player_center()
hole_size = [player.L, player.L]



player_won = False
finished = -1
clock = pygame.time.Clock()
done = False
while not done:
    clock.tick(FPS)
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            done = True
    if finished < 0:
        if player.get_altitude() < 0 or player_won:
            finished = iteration
    elif finished > 0 and iteration - finished > 3*FPS:
        done = True
    mon.append("a")
    player.update_command()
    player.update_physics()
    player_center = get_player_center()
    draw()
    draw_vect(screen, player_center, (0,0,0), player.vel)
    draw_vect(screen, player_center, (0,0,255), player.orientation*100)
    pygame.display.flip()
    if player_won:
        if iteration - finished < FPS//8:
            make_debris_explosion()
    elif player.is_dead():
        if iteration - finished < FPS//8:
            make_debris_explosion()
        player.vel *= 0.1
##        iteration = 10**5 - 10
##        done = True
    iteration += 1
##    print(debrisgen3.debris)
    if iteration%100 == 0:
        mon.show()


app.quit()

#TODO: Generation monde infini
#TODO: GUI + vitesse rouge clignotant quand approche vitesse décrochage
#TODO: sons : alarme altimetre, vent (avec volume = vitesse), holes, explosion, booster
#TODO: moments de reparation/changements
#TODO: fumée si avion trop endommagé.
#TODO: moins aligner planeur a vitesse : comment faire un avion qui plane mal mais a peu de trainée ? Le fer a repasser tourne mais est moins sur des rails.

#TODO: remettre systeme holes pour utiliser obstacles prefabriqués ? Oui SI BESOIN car:
    #permet de reduire temps de chargement
    #permet de faire de + grandes surfaces
    #checker monitoring.
