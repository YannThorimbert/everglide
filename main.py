import pygame, thorpy
from pygame.math import Vector2 as V2
import random
import math

W,H = 1200,700
FPS = 80
DT = 0.08
GRAVITY = 9.81
SPACE_ALTITUDE = 3000
CPT_GAP = 40000 #1000 = 100m
COLOR_OBSTACLE = (100,)*3
OBS_W = 150
OBS_SPACE = 50
HMIN, HMAX = 50, 150 #obstacle random height borns
PLAYER_CENTER0 = 300, H//2
JERRICAN_FUEL = 0.2
FONT = 'PressStart2P-Regular.ttf'
W_GUI, H_GUI = 200, 30

def draw_vect(surface, p0, color, v):
    pygame.draw.line(surface, color, p0, p0 + v, 3)

def rand_sgn():
    if random.random() < 0.5:
        return 1
    return -1

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
    def __init__(self, img):
        #aerodynamics
        self.scx = 1.
        self.scz = 1.
        self.scx_90 = 3.
        self.k = 0.1 #railing
        #misc specs
        self.price = 0
        self.fuel = 1.
        self.power = 10000.
        self.consumption = 3e-3
        self.command_torque = 3.
        self.throttling = False
        self.damage = 0.
        self.fragility = 1e-3
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
        #image
        self.img = img
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
        a_rad = a * math.pi / 180.
        return self.scx + self.scx_90*math.sin(a_rad)

    def get_command_torque(self):
        v = self.vel.length()
        if v > self.vel_stall:
            return self.command_torque
        return self.command_torque * ( 1. - v/self.vel_stall)

    def get_altitude(self):
        return -self.pos.y

    def get_distance(self):
        return int(self.pos.x/10.)

    def update_physics(self):
        #update angle of attack
        self.aoa = to_small_form(self.vel.angle_to(self.orientation))
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
        if keys[pygame.K_RIGHT] or keys[pygame.K_UP]:
            self.torque += self.get_command_torque()
        if keys[pygame.K_LEFT] or keys[pygame.K_DOWN]:
            self.torque -= self.get_command_torque()
        if keys[pygame.K_SPACE]:
            if self.fuel > 0:
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
        return self.get_altitude() < 0 or self.damage >= 1.


def make_debris():
    angle = 0
    spread = 180 #spread of debris directions
    vel = int(player.vel.length()*DT)
    debrisgen.generate( player_center, #position
                        n=2, #number of debris
                        v_range = (vel, 4*vel),
                        omega_range=(5,25), #rotational velocity range
                        angle_range=(angle-spread,angle+spread))

def make_debris_explosion():
    angle = 180
    spread = 30 #spread of debris directions
    vel = 40
    debrisgen2.generate( player_center, #position
                        n=40, #number of debris
                        v_range = (vel, 4*vel),
                        omega_range=(5,25), #rotational velocity range
                        angle_range=(angle-spread,angle+spread))

def make_coin_explosion():
    angle = 180
    spread = 180 #spread of debris directions
    vel = 20
    debrisgen3.generate( player_center, #position
                        n=100, #number of debris
                        v_range = (vel, 4*vel),
                        omega_range=(5,25), #rotational velocity range
                        angle_range=(angle-spread,angle+spread))


def draw_gui():
    x,y = 50, 10
    dy = 30
    #Coins
    img_boost.fill((255,255,255))
    rboost = img_boost.get_rect()
    r = coin_img_gui.get_rect()
    r.centery = rboost.centery
    r.x = rboost.x + 2
    img_boost.blit(coin_img_gui, r.topleft)
    r2 = e_txt_coin.get_fus_rect()
    r2.centery = rboost.centery
    r2.x = r.right + 10
    e_txt_coin.set_topleft(r2.topleft)
    e_txt_coin.set_text(str(player_money))
    img_boost.blit(e_txt_coin.get_image(), r2.topleft)
    screen.blit(img_boost, (x,y))
    y += dy + 10
    #Fuel
    img_boost.fill((255,255,255))
    r = img_boost.get_rect()
    if player.fuel < 0.2:
        color = (255-4*(iteration%63),255-4*(iteration%63),0)
        pygame.draw.rect(img_boost, color, r, 5)
    r.inflate_ip((-2,-2))
    rtxt = txt_boost.get_rect()
    rtxt.center = r.center
    r.w = player.fuel * r.w
    f = max(player.fuel, 0)
    pygame.draw.rect(img_boost, (255*(1.-f),255*f,0), r)
    img_boost.blit(txt_boost, rtxt.topleft)
    screen.blit(img_boost, (x,y))
    y += dy + 10
    #
##    img_boost.fill((255,255,255))
##    r = img_boost.get_rect().inflate((-2,-2))
##    rtxt = txt_vel.get_rect()
##    rtxt.center = r.center
##    V = player.vel.length()
##    r.w = min(V, wgui)
##    pygame.draw.rect(img_boost, (255,0,0), r)
##    img_boost.blit(txt_vel, rtxt.topleft)
##    screen.blit(img_boost, (x,y))
##    y += dy + 10
    #Damage
    img_boost.fill((255,255,255))
    r = img_boost.get_rect()
    if player.damage > 0.8:
        color = (255-4*(iteration%63),255-4*(iteration%63),0)
        pygame.draw.rect(img_boost, color, r, 5)
    r.inflate_ip((-2,-2))
    rtxt = txt_damage.get_rect()
    rtxt.center = r.center
    V = player.vel.length()
    r.w = (1. - player.damage) * r.w
    pygame.draw.rect(img_boost, (255,0,0), r)
    img_boost.blit(txt_damage, rtxt.topleft)
    screen.blit(img_boost, (x,y))
    y += dy + 10
    #texts
    e_distance.set_text(str(player.get_distance())+" m")
    e_distance.blit()
    e_cpt.blit()
    e_hiscore.blit()

def draw_sky(altitude):
    f = 1. - altitude / SPACE_ALTITUDE
    if f > 1:
        f = 1
    elif f < 0:
        f = 0
    screen.fill((150*f,150*f,255*f))
    star.set_alpha((1-f)*255)
    screen.blits(stars)
    moon.set_alpha((1-f)*250 + 5)
    screen.blit(moon, (W-200,80))
    return f

def draw_far_field(v):
    for img, pos in far_field2:
        screen.blit(img, pos)
        pos[1] -= v.y/ 100.
    for pos,img in far_field:
        screen.blit(img, pos)
        pos[0] -= v.x / 30.
        if pos[0] < -300:
            pos[0] += W + 300
        pos[1] -= v.y / 30.

def reset_far_field():
    for img, pos in far_field2:
        h = img.get_height()
        pos[1] = H - h + 100
    for pos,img in far_field:
        h = img.get_height()
        pos[1] = H - h + 100

def set_trend(what):
    global obs_shift_sgn, obs_shift_intensity, jerrican_probability
    if what == "descent":
        obs_shift_intensity = random.choice([0, 100, 200, 300, 600])
        jerrican_probability = 0.
        obs_shift_sgn = -1
    elif what == "climb":
        obs_shift_intensity = random.choice([0, 100, 200, 300, 500])
        jerrican_probability = obs_shift_intensity/2000. + 0.1
        obs_shift_sgn = 1
    else:
        assert False
    d = player.get_distance()
    if d > 25000:
        factor = (50000-d)/25000
        if factor < 0.1:
            factor = 0.1
        jerrican_probability *= factor

def refresh_draw_obstacles(f, player_velnorm):
    global obs_shift
    obs_shift += obs_shift_sgn * obs_shift_intensity * 3e-3
    if obs_shift > SPACE_ALTITUDE-H//2:
        set_trend("descent")
    elif obs_shift < 0:
        set_trend("climb")
    elif random.random() < 1e-4:
        if random.random() < 0.5:
            set_trend("descent")
        else:
            set_trend("climb")
    for i, data in enumerate(obstacles):
        rect, img = data
        x,y = rect.topleft
        o_rect = img.get_rect()
        o_rect.topleft = x-player.pos.x, y-player.pos.y
        if o_rect.x < -2*OBS_W :
            rect.x += len_obs
            draw_coin[i] = True
            coins[i].x = rect.x + random.randint(-100,100)
            jerricans[i].x = rect.x + random.randint(-100,100)
            new_h = random.choice(possible_h)
            obstacles[i][1] = precomp_rects[new_h]
            rect.y = altitude_to_y(new_h) - obs_shift + random.randint(-100,100)
            if random.random() < 0.2:
                rect.y -= random.randint(200,600)
            if rect.y > new_h:
                rect.y = new_h
            coins[i].y = rect.y - random.randint(20,100)
            jerricans[i].y = rect.y - random.randint(80,150)
            draw_jerrican[i] = random.random() < jerrican_probability
            if not draw_jerrican[i]:
                if random.random() < 0.1:
                    jerricans[i].y = -200
                    draw_jerrican[i] = True
            rect.h = new_h
            continue
        screen.blit(img, o_rect)
        if finished < 0:
            o_rect.topleft = x-player_center[0], y-player_center[1]
            for point in player.iterate_points():
                collision = False
                if o_rect.collidepoint(point):
                    mod = int(500. / (player_velnorm+0.1))
                    if iteration%mod == 0:
                        r = pygame.Rect((0,0),hole_size)
                        r.center = player_center
                        holes.append([r.x+player.pos.x,r.y+player.pos.y])
                    if random.random() < 0.5:
                        make_debris()
                        sm.explosion.stop()
                        sm.explosion.play()
                    player.vel *= 0.995
                    player.damage += player.fragility
                    collision = True
                    break
                if collision:
                    break

def draw_coins():
    global player_money
    for i, pos in enumerate(coins):
        rect = coin_img.get_rect()
        rect.topleft = pos.x-player.pos.x, pos.y-player.pos.y
        if draw_coin[i]:
            screen.blit(coin_img, rect)
            rect.topleft = pos.x-player_center[0], pos.y-player_center[1]
            for point in player.iterate_points():
                if rect.collidepoint(point):
                    draw_coin[i] = False
                    player_money += 1
                    sm.coin.play()
                    break

def draw_jerricans():
    rect_jerrican = jerrican_img.get_rect()
    for i, pos in enumerate(jerricans):
        x_img = pos.x - player.pos.x
        y_img = pos.y - player.pos.y
        if draw_jerrican[i]:
            screen.blit(jerrican_img, (x_img,y_img))
            rect_jerrican.topleft = pos.x - player_center[0], pos.y - player_center[1]
            for point in player.iterate_points():
                if rect_jerrican.collidepoint(point):
                    sm.fuel.play()
                    draw_jerrican[i] = False
                    player.fuel += JERRICAN_FUEL
                    if player.fuel > 1:
                        player.fuel = 1
                    break

def treat_damage_and_blit_player(player_velDT):
    global alarm_played
    if finished < 0 or transitioning:
        screen.blit(player.img, player_rect)
    elif player_won:
        screen.blit(player.img, player_rect)
        debrisgen3.kill_translate_update_draw(screen_rect, V2(0,4), DT, screen)
    elif not transitioning:
        debrisgen2.kill_translate_update_draw(screen_rect, V2(0,4), DT, screen)
    else:
        screen.blit(player.img, player_rect)
    debrisgen.kill_translate_update_draw(screen_rect, -player_velDT + V2(0,2),
                                            DT, screen)
    smokegen2.kill_update_draw(-V2(player_velDT), screen)
    if player.damage > 0.8:
        if not alarm_played:
            sm.alarm_damage.play(5)
        alarm_played = True
        smokegen.generate(V2(player_center)+(0,10))
        smokegen.kill_update_draw(-V2(player_velDT), screen)
    else:
        sm.alarm_damage.stop()

def draw_wind(player_velnorm, player_velDT):
    angle = player.orientation.angle_to(V2(1,0))
    wind_img = pygame.transform.rotate(wind_img_orig, angle)
    wind_img2 = pygame.transform.rotate(wind_img_orig2, angle)
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

def refresh_game():
    global done, holes, player_money, x_grass
    global player_rect, player_center
    f_sin = 0.5*math.sin(iteration*0.1) + 0.5
    player_velnorm = player.vel.length()
    player_velDT = player.vel*DT
    #Refresh wind sounds
    wind_volume = min(1, player_velnorm/100.)**2
    if wind_volume < 0:
        wind_volume = 0
    sm.wind.set_volume(wind_volume)
    #Refresh player image and box
    player.align_img_to_orientation()
    player_rect = player.img.get_rect()
    player_center = get_player_center()
    player_rect.center = player_center
    #Draw sky ##################################################################
    f = draw_sky(player.get_altitude())
    #Draw far field ############################################################
    draw_far_field(player_velDT)
    #Draw checkpoint line ######################################################
    if player.pos.x > 100:
        x = next_cpt_x - player.pos.x + player_center[0]
        pygame.draw.rect(screen, (255,255*f_sin,0), pygame.Rect(x, 0, 200, H))
    #Draw objects ##############################################################
    refresh_draw_obstacles(f,player_velnorm)
    L = len(holes)
    if L > MAX_HOLES:
        holes = holes[L-MAX_HOLES:]
    s_hole.fill((150*f,150*f,255*f))
    for x,y in holes:
        screen.blit(s_hole,(x-player.pos.x, y-player.pos.y))
    draw_coins()
    draw_jerricans()
    #Draw ground ###############################################################
    y_ground = altitude_to_y(0) - player.pos.y
    pygame.draw.rect(screen, grass_color, pygame.Rect(0,y_ground+128,W,H//2-128))
    screen.blit(grass_img, (x_grass,y_ground))
    screen.blit(grass_img, (x_grass+W,y_ground))
    x_grass -= player_velDT.x
    if x_grass < -W:
        x_grass = 0
    treat_damage_and_blit_player(player_velDT)
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
    #Draw wind #################################################################
    if player_velnorm > 40:
        draw_wind(player_velnorm, player_velDT)
    #Draw altitude alert #######################################################
    if H//2 < player.get_altitude() < 1.5*H:
        if player.vel.y > 20.:
            ground_alert.set_alpha(f_sin*255)
            screen.blit(ground_alert, ((W-ground_alert.get_width())//2, H-60))

def transition_to(y):
    N = 200
    delta = (y - player.pos.y) / N
    for i in range(N):
        player.pos.y += delta
        refresh_game()
        pygame.display.flip()

def add_obstacle(r):
    img = precomp_rects[r.h]
    pygame.draw.rect(img, (0,0,0), img.get_rect(), 3)
    r.topleft = r.x, altitude_to_y(r.y)
    obstacles.append([r,img])

def altitude_to_y(a):
    return H//2 - a


def get_player_center():
    x0, y0 = PLAYER_CENTER0
    f = 2.
    return x0 + player.vel[0]/f, y0 + player.vel[1]/f

def initialize_obstacles():
    global len_obs
    prev = 300. #START POS
    while True:
        h = random.choice(possible_h)
        r = pygame.Rect(prev, h, OBS_W, h)
        add_obstacle(r)
        prev = r.right + OBS_SPACE
        if prev > W + 4*OBS_W:
            break
    len_obs = len(obstacles)*(OBS_W+OBS_SPACE)

def pause_menu():
    sm.wind.set_volume(0.)
    def launch_credits():
        cb = thorpy.make_textbox("Credits",
            "Game created by Yann Thorimbert for the Pyweek challenge.\n"+\
            "yann.thorimbert@gmail.com")
        cb.center()
        thorpy.launch_blocking(cb)
        refresh_game()
        b.blit()
        pygame.display.flip()
    e_credits_button = thorpy.make_button("Credits", launch_credits)
    b = thorpy.Box([e_goal,e_commands, e_credits_button, e_quit, e_ok])
    b.center()
    b.set_main_color((200,200,200,200))
    thorpy.launch_blocking(b)

def countdown(t):
    clock = pygame.time.Clock()
    done = False
    e = thorpy.Element("Start in "+str(round(t/1000.)) + " s")
    e.set_font_size(30)
    e.set_main_color((255,255,0))
    e.scale_to_title()
    e.center()
    tot_time = t
    while tot_time > 0:
        tot_time -= clock.tick(FPS)
        e.blit()
        e.update()
        e.set_text("Start in "+str(round(tot_time/1000.)) + " s")

def add_far_field(x,w,h,c,k):
    if k != 0:
        h += 100
    s = pygame.Surface((w,h+200))
##    s.fill((c,)*3)
    s.fill((0,0,c))
    if k == 0:
        far_field.append(([x,H-h],s))
    else:
        far_field2.append((s,[x,H-h+100]))



def cs_time_event():
    global cs_it, cs_sgn
    draw_sky(cs_it*2)
    cs_it += cs_sgn*20.
    if cs_it > SPACE_ALTITUDE:
        cs_sgn = -1
    elif cs_it < 0:
        cs_sgn = 1
    draw_far_field(V2())
    title.blit()
    cs.blit()
    pygame.display.flip()

def plane_choice():
    sm.wind.set_volume(0.)
    global player_money
    button_planes = []
    class Choice:
        choice = None
    def choose(value):
        global player_money
        if player_money >= value.price:
            player_money -= value.price
            Choice.choice = value
        thorpy.functions.quit_menu_func()
    for p in planes:
        img = p.img_orig.copy()
        img.convert()
        img.set_colorkey((0,0,0))
        img = thorpy.Image(img)
        infos = thorpy.make_text(p.infos_txt, 12)
        if player_money >= p.price:
            b = thorpy.Clickable(elements=[img, infos])
            b.user_func = choose
            b.user_params = {"value":p}
        else:
            b = thorpy.Element(elements=[img, infos])
            b.set_pressed_state()
            b.set_active(False)
        thorpy.store(b, [img, infos])
        b.fit_children()
        button_planes.append(b)
    g1 = thorpy.make_group(button_planes[0:2])
    g2 = thorpy.make_group(button_planes[2:4])
    title = thorpy.make_text("      You reached a checkpoint.\n"+\
                                "Do you want to buy a new plane ?",
                                30, (255,0,0))
    nothx = thorpy.make_button("No, thanks (press <Enter>)", thorpy.functions.quit_menu_func)
    nothx.set_font_size(20)
    nothx.scale_to_title()
    b = thorpy.Box([title,nothx,g1,g2])
    b.set_main_color((200,200,200,200))
    b.center()
    thorpy.launch_blocking(b,add_ok_enter=True)
    return Choice.choice
#End of functions ##############################################################
#precompute obstacles images
possible_h = list(range(HMIN,HMAX,20))
precomp_rects = {}
for h in possible_h:
    s = pygame.Surface((OBS_W, h))
    s.fill(COLOR_OBSTACLE)
    pygame.draw.rect(s, (0,0,0), s.get_rect(), 3)
    precomp_rects[h] = s

#global variables
hi_score = 0
last_won = 0
transitioning = False
player_won = False
jerrican_probability = 0.1
player_money = 0
obs_shift = 0
obs_shift_sgn = -1
obs_shift_intensity = 10
alarm_played = False
obstacles = []
len_obs = -1
initialize_obstacles()
coins = [V2(r.topleft) - (random.randint(-100,100),random.randint(50,100)) for r,img in obstacles]
draw_coin = [True for c in coins]
jerricans = [V2(r.topleft) - (random.randint(-100,100),random.randint(50,100)) for r,img in obstacles]
draw_jerrican = [False for j in jerricans]
thorpy.set_default_font(FONT)
thorpy.style.FONT_SIZE = 20
#create app
app = thorpy.Application((W, H), "Everglide")
thorpy.set_theme("simple")
screen = app.get_screen()
screen_rect = screen.get_rect()
e_goal = thorpy.make_textbox_nobutton("Goal",
        "Go as far as possible before crashing.\n"+\
        "Collect coins to buy new planes.\n"+\
        "Jerricans are often close to the coins\n"
        "Try to beat the score of 25'000 m.\n"+\
        "Press <Space> to boost your glider.\n"+\
        "The walls cause damage and slow down your glider.")
e_commands = thorpy.make_textbox_nobutton("Commands",
    "Turn: <Left>/<Right>: <Space>")
e_goal_commands = thorpy.make_group([e_goal, e_commands], "v")
e_goal_commands.center()
e_quit = thorpy.make_button("Quit game", thorpy.functions.quit_func)
e_quit.set_main_color((200,0,0))
e_ok = thorpy.make_button("Ok", thorpy.functions.quit_menu_func)
e_ok.set_main_color((0,200,0))

#Sky and ground elements
ground_alert = pygame.Surface((3*W//4, 40))
ground_alert.fill((255,0,0))
winds = [V2(random.randint(0,W), random.randint(0,H)) for i in range(20)]
star = pygame.Surface((3,3))
star.fill((255,255,0))
star.set_alpha(0)
stars = []
for i in range(30):
    x = random.randint(0,W)
    y = random.randint(0,2*H//3)
    stars.append((star, (x,y)))
moon = pygame.transform.scale(pygame.image.load("moon.png"), (16, 16)).convert()
moon.set_colorkey((0,0,0))
moon = pygame.transform.scale(moon, (100,100))
clouds = []
for i in range(10):
    x = random.randint(0,5000)
    y = random.randint(-2*H, 0)
    w = random.randint(50, 300)
    h = random.randint(50, w)
    img = pygame.Surface((w,h))
    c = random.choice([220,250])
    img.fill((c,)*3)
    img.set_alpha(127)
    dx = random.choice([-2,-1])
    clouds.append((V2(x,y), dx, img))
wind_img_orig = pygame.Surface((20,20))
pygame.draw.rect(wind_img_orig, (255,255,255), pygame.Rect(0,9,20,3))
wind_img_orig.set_colorkey((0,0,0))
wind_img_orig2 = pygame.transform.scale(wind_img_orig, (10,10))
#create far field image
far_field = []
far_field2 = []
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
jerrican_img = pygame.transform.scale(pygame.image.load("jerrican.png"), (40,40))
jerrican_img.set_colorkey((255,)*3)
coin_img = pygame.transform.scale(pygame.image.load("coin.png"), (40,40)).convert()
coin_img.set_colorkey((255,)*3)
coin_img_gui = pygame.transform.scale(coin_img, (20,20))
grass_tile = pygame.transform.scale(pygame.image.load("grass.png"), (128,128))
grass_img = pygame.Surface((W,128))
##grass_color = (77,49,9)
grass_color = grass_tile.get_at((64,127))
x_grass = 0
while x_grass < W:
    grass_img.blit(grass_tile, (x_grass,0))
    x_grass += 128
x_grass = 0

#Build ingame GUI elements
img_boost = pygame.Surface((W_GUI, H_GUI))
img_boost.set_alpha(150)
TXT_COLOR = (180,)*3
txt_boost = thorpy.make_text("Fuel", 20, TXT_COLOR).get_image()
txt_vel = thorpy.make_text("Speed", 20, TXT_COLOR).get_image()
txt_damage = thorpy.make_text("Damage", 20, TXT_COLOR).get_image()
e_txt_coin = thorpy.make_text("0", 20, TXT_COLOR)
e_distance = thorpy.make_text("0000 m", 30, (255,255,0))
e_distance.stick_to("screen", "top", "top")
e_distance.move((0,10))
e_next_cpt = thorpy.make_text("Next checkpoint:", 20, (255,255,0))
e_next_cpt_dist = thorpy.make_text("0000", 20, (255,255,0))
e_cpt = thorpy.make_group([e_next_cpt, e_next_cpt_dist])
e_cpt.stick_to(e_distance, "bottom", "top")
e_hiscore = thorpy.make_text("Hi score: 0 m", 20, (255,255,0))
e_hiscore.stick_to("screen", "right", "right", align=False)
e_hiscore.move((0,10))

title = thorpy.make_text("Everglide", 50, (255,255,0))
title.stick_to("screen","top","top")
title.move((0,50))

cs = thorpy.ColorSetter(color_size=(200,200),
                        value=(50,50,50),
                        elements=[thorpy.make_text("Choose your plane color", 30),
                        thorpy.make_text("(press <Enter> to validate)", 20),
                        thorpy.make_text("  ", 30)])
cs.center()
cs.set_value((255,0,0))
cs_it = 0
cs_sgn = 1
thorpy.add_time_reaction(cs, cs_time_event)
draw_sky(0.)
draw_far_field(V2())
pygame.display.flip()
c = thorpy.launch_blocking(cs)
player_color = cs.get_color()
if player_color == (255,)*3:
    player_color = (253,)*3
draw_sky(cs_it)
draw_far_field(V2())
e_goal_commands.blit()
e_start = thorpy.make_text("Loading the world...", 20, (255,255,0))
e_start.stick_to(e_goal_commands, "bottom", "top")
e_start.move((0,20))
e_start.blit()
pygame.display.flip()

#Build particles. This is the slow part of the loading process.
smokegen = thorpy.fx.get_smokegen(n=30, color=(200,200,200), grow=0.4, size0=(15,15))
smokegen2 = thorpy.fx.get_fire_smokegen(n=40, color=(200,255,155), grow=0.8, size0=(10,10))
debrisgen = thorpy.fx.get_debris_generator(duration=200, #nb. frames before die
                                    color=COLOR_OBSTACLE,
                                    max_size=20)
debrisgen2 = thorpy.fx.get_debris_generator(duration=1000, #nb. frames before die
                                    color=player_color,
                                    max_size=20)
debrisgen3 = thorpy.fx.get_debris_generator(duration=1000, #nb. frames before die
                                    color=(255,255,0),
                                    max_size=20,
                                    samples=[coin_img, coin_img])
#Build planes
all_planes = pygame.image.load("planes.png")
thorpy.change_color_on_img_ip(all_planes, (255,255,0), player_color)
thorpy.change_color_on_img_ip(all_planes, (224,224,0), tuple([0.6*c for c in player_color]))
img_planes = []
for i in range(all_planes.get_width()//32):
    s = pygame.Surface((32,32))
    s.fill((255,)*3)
    s.blit(all_planes, (-i*32,0))
    s.set_colorkey((255,)*3)
    img_planes.append(s)
plane0 = Plane(img_planes[0])
plane0.infos_txt = "0 coin\n\nSpeed      **\nHandling   ****\nFuel       ***\nRobustness **"
plane0.price = 0
plane0.scx = 0.03
plane0.scx_90 = 2.
plane0.scz = 0.2
plane0.mass = 900.
plane0.I = 0.1
plane0.consumption = 3e-3
plane0.fragility = 2e-3

plane1 = Plane(img_planes[1])
plane1.infos_txt = "30 coins\n\nSpeed      ****\nHandling   *\nFuel       ****\nRobustness *****"
plane1.price = 30
plane1.scx = 0.035
plane1.scx_90 = 3.
plane1.scz = 0.1
plane1.I = 0.15
plane1.k = 0.035
plane1.mass = 2000.
plane1.consumption = 3e-3
plane1.power = 40000
plane1.fragility = 5e-4

plane2 = Plane(img_planes[3])
plane2.infos_txt = "30 coins\n\nSpeed      ***\nHandling   ****\nFuel       **\nRobustness ***"
plane2.price = 30
plane2.scx = 0.02
plane2.scx_90 = 3.
plane2.scz = 0.19
plane2.I = 0.15
plane2.k = 0.08
plane2.mass = 1100
plane2.consumption = 3e-3
plane2.power = 17000
plane2.fragility = 2e-3

plane3 = Plane(img_planes[2])
plane3.infos_txt = "80 coins\n\nSpeed      *\nHandling   *****\nFuel       ****\nRobustness *"
plane3.price = 80
plane3.scx = 0.03
plane3.scx_90 = 2.
plane3.scz = 0.3
plane3.I = 0.08
plane3.k = 0.2
plane3.mass = 800.
plane3.consumption = 2e-3
plane3.power = 7000
plane3.fragility = 4e-3

planes = [plane0, plane1, plane2, plane3]
finished = -1
e_start.set_text("Press <Enter> to start")
draw_sky(cs_it)
draw_far_field(V2())
e_goal_commands.blit()
e_start.blit()
pygame.display.flip()
app.pause_until(pygame.KEYDOWN, key=pygame.K_RETURN)
#Initialize player
H0 = 400
player = plane0
player.pos.y = obstacles[0][0].top - H//2 - H0
player.pos.x = -300
player.vel = V2(100, 0)
player.orientation = player.vel.normalize()
iteration = 0
next_cpt_x = CPT_GAP
e_next_cpt_dist.set_text(str(int(next_cpt_x/10.))+" m")
holes = []
hole_size = [player.L, player.L]
s_hole = pygame.Surface(hole_size)
MAX_HOLES = 2 * W//hole_size[0]

#Load sounds
sm = thorpy.SoundCollection()
sm.add("sketchman3.ogg", "wind")
sm.add("alarm.ogg", "alarm_damage")
sm.alarm_damage.set_volume(0.06)
sm.add("sfx_explosionNormal.ogg", "explosion")
sm.explosion.set_volume(0.07)
sm.add("SoundExplorer.wav", "explosion2")
sm.explosion2.set_volume(0.5)
sm.add("coin.wav", "coin")
sm.coin.set_volume(0.5)
sm.add("ok.wav", "fuel")
sm.fuel.set_volume(0.5)
sm.add("OpenSurge.ogg", "jingle_won")
sm.jingle_won.set_volume(0.25)
sm.add("spuispuin.wav", "jingle_change")
sm.wind.play(-1)
sm.wind.set_volume(0.)

countdown(3000)
clock = pygame.time.Clock()
done = False
while not done:
    clock.tick(FPS)
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            done = True
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                pause_menu()
            elif e.key == pygame.K_PRINT:
                pygame.image.save(screen, "screenshot.png")
    if finished < 0:
        if player.is_dead() or player_won:
            if not player_won:
                sm.explosion2.play()
            finished = iteration
    elif finished > 0 and iteration - finished > 1*FPS:
        debrisgen.debris = []
        debrisgen2.debris = []
        debrisgen3.debris = []
        alarm_played = False
        sm.alarm_damage.stop()
        reset_far_field()
        if player_won:
            new_player = plane_choice()
            if new_player:
                new_player.pos = player.pos
                player = new_player
        else:
            obstacles = []
            initialize_obstacles()
            player_distance = player.get_distance()
            if hi_score < player_distance:
                sm.jingle_won.play()
                hi_score = player_distance
                e_hiscore.set_text("Hi score: "+str(hi_score)+" m")
                e_hiscore.stick_to("screen", "right", "right", align=False)
                if hi_score > 25000:
                    e_txt = thorpy.make_text("Congratulations !\nThis is an amazing score.\n"+\
                    "You could leave without regrets now...", 30, (255,0,0))
                else:
                    e_txt = thorpy.make_text("New top score: " + str(hi_score)+\
                        " m\nTry to improve your score.\nLegendary score is 25'000 m.", 30, (255,0,0))
            else:
                e_txt = thorpy.make_text("Retry !\nYour current top score is " + str(hi_score)+" m", 30, (255,0,0))
            e_ok = thorpy.make_button("Ok", thorpy.functions.quit_menu_func)
            e_ok.set_font_size(30)
            e_ok.scale_to_title()
            box = thorpy.Box([e_txt, e_ok])
            box.set_main_color((200,200,200,127))
            box.center()
            thorpy.launch_blocking(box)
            player_money = 0
            player = plane0
            player.pos.x = 0
            obs_shift = 0
            obs_shift_sgn = -1
            obs_shift_intensity = 10
            holes = []
            transitioning = True
            transition_to(obstacles[0][0].top - H//2 - H0)
        player_won = False
        player.fuel = 1.
        player.damage = 0.
        player.vel = V2((100,0))
        player.orientation = player.vel.normalize()
        transitioning = False
        finished = -1
        sm.jingle_won.stop()
        countdown(3000)
    player.update_command()
    player.update_physics()
    refresh_game()
    draw_gui()
    pygame.display.flip()
    if player_won:
        if iteration - finished < FPS//8:
            make_debris_explosion()
    elif player.is_dead():
        if iteration - finished < FPS//8:
            make_debris_explosion()
        player.vel *= 0.1
    iteration += 1
    if not(player_won) and player.pos.x > 100:
        if next_cpt_x != last_won:
            if player.pos.x > next_cpt_x:
                sm.jingle_change.play()
                player_won = True
                make_coin_explosion()
                last_won = next_cpt_x
                next_cpt_x += CPT_GAP
                e_next_cpt_dist.set_text(str(int(next_cpt_x/10.))+" m")
app.quit()
