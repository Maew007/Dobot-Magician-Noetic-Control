#!/usr/bin/env python3
"""
dobot_button_gui.py — Tk GUI controller for DOBOT Magician sim (ROS).

Features:
  Motion tab
    - Pose presets: HOME / READY / PICK / PLACE (smooth 1 s)
    - Jog buttons (auto-repeat on hold):
        LEFT/RIGHT -> joint_1 (base yaw)
        UP/DOWN    -> joint_5 (forearm pitch)
    - Gripper OPEN/CLOSE
    - Per-joint sliders (instant follow) + - / + buttons (auto-repeat)
    - Step size radios 0.01 / 0.05 / 0.1 / 0.5 rad
    - STOP / HOME / Sync Sliders / Quit
  Sequence tab
    - Treeview: # / Name / J1 / J2 / J5 / J6 / J7 / Dwell
    - Add (Current), Add (Preset), **Go to Step** (or click row),
      Capture Pose, Edit Name/Dwell, Delete, Up, Down
    - Play (smooth + dwell) / Stop / Save JSON / Load JSON / Clear

Publishes /joint_states (sensor_msgs/JointState) on 5 joints matching the
project URDF:  joint_1, joint_2, joint_5, joint_6, joint_7.

Run via:
    roslaunch magician_control dobot_gui.launch
"""

import os
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import rospy
from sensor_msgs.msg import JointState

# ----- Configuration ------------------------------------------------------

JOINT_NAMES = ['joint_1', 'joint_2', 'joint_5', 'joint_6', 'joint_7']

# Per-joint limits (rad), from URDF revolute limit tags.
JOINT_LIMITS = [
    (-3.14159, 3.14159),    # joint_1
    ( 0.0,     1.57079),    # joint_2: 0..pi/2
    ( 0.0,     1.57079),    # joint_5: 0..pi/2
    (-3.14159, 3.14159),    # joint_6
    (-3.14159, 3.14159),    # joint_7
]

PRESETS = {
    'HOME':       [ 0.00,  0.00,  0.50,  0.00,  0.00],
    'READY':      [ 0.00,  0.30,  0.40,  0.00,  0.00],
    'PICK':       [ 0.00,  0.40,  0.70, -0.40,  0.00],
    'PLACE':      [ 0.00,  0.70,  0.50,  0.00,  0.00],
    'GRIP_OPEN':  [ 0.00,  0.00,  0.50,  0.00,  0.00],
    'GRIP_CLOSE': [ 0.00,  0.00,  0.50,  0.00, -0.50],
}

JOG_LEFT  = 0; JOG_RIGHT = 0
JOG_UP    = 2; JOG_DOWN  = 2

INTERP_PRESET = 1.0          # sec for preset-button smoothing
INTERP_JOG    = 0.25         # sec for jog-button smoothing
INTERP_STEP   = 0.30         # sec for +/- smoothing
INTERP_SEQ    = 1.0          # sec for sequence-step smoothing
PUBLISH_RATE_HZ = 20
POLL_HZ         = 10
HOLD_DELAY_MS   = 350
HOLD_REPEAT_MS  = 60
N = len(JOINT_NAMES)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class DobotGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("DOBOT Magician - Sim Controller")
        self.root.geometry("960x780")

        rospy.init_node("dobot_button_gui", anonymous=True)
        self.js_pub = rospy.Publisher('/joint_states', JointState, queue_size=5)

        self.lock = threading.Lock()
        self.target = list(PRESETS['HOME'])
        self.current = list(PRESETS['HOME'])
        self.last_target = list(PRESETS['HOME'])
        self.last_target_time = time.time() - INTERP_PRESET
        self.current_duration = INTERP_PRESET
        self.running = True

        # When True, the publisher's slider-snap is suppressed so a sequence
        # step's smooth interpolation can run without being instantly overridden.
        self.suppress_slider_snap = False
        # Skip the next N publisher snap checks (used by _seq_command_to to
        # absorb the race between slider-var read and click-handler update).
        self.skip_snap_count = 0
        # Pending Tk updates from non-main threads (e.g. seq_run, click handler).
        # Tick on main thread via _tick_gui — keeps ALL Tk IPC on main thread
        # so we never queue hundreds of DoubleVar.set() calls from non-main.
        self.pending_slider = None
        self.pending_status = None

        self.sequence = []
        self.playing = False
        self.play_stop = threading.Event()

        self.step_size = tk.DoubleVar(value=0.1)
        self.slider_vars = [tk.DoubleVar(value=PRESETS['HOME'][i]) for i in range(N)]
        # When True, _seq_run replays the whole sequence after it finishes
        # until either Stop is pressed or the checkbox is unchecked.
        self.loop_sequence = tk.BooleanVar(value=False)

        # ---- Programs tab state ----
        # A 'program' is one saved JSON sequence loaded from disk. The user
        # composes several programs into a master list and Play All runs
        # them back-to-back (with optional looping).
        self.programs = []           # list of {name, path, steps}
        self.prog_playing = False
        self.prog_stop = threading.Event()
        self.loop_programs = tk.BooleanVar(value=False)

        self._build_widgets()
        self.root.after(50, self._tick_gui)
        threading.Thread(target=self._publisher_loop, daemon=True).start()
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    # ----- Button factory with hold-to-repeat ------------------------------

    def _make_hold_button(self, parent, text, command, **kw):
        btn = ttk.Button(parent, text=text, **kw)
        state = {'after_id': None}

        def _start(e=None):
            if state['after_id'] is None:
                command()
                state['after_id'] = btn.after(HOLD_DELAY_MS, _tick)

        def _tick():
            if state['after_id'] is None:
                return
            command()
            state['after_id'] = btn.after(HOLD_REPEAT_MS, _tick)

        def _stop(e=None):
            if state['after_id'] is not None:
                btn.after_cancel(state['after_id'])
                state['after_id'] = None

        btn.bind('<ButtonPress-1>', _start, add='+')
        btn.bind('<ButtonRelease-1>', _stop, add='+')
        btn.bind('<Leave>', _stop, add='+')
        return btn

    # ----- Layout ---------------------------------------------------------

    def _build_widgets(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=8, pady=8)

        self.tab_motion = ttk.Frame(nb)
        nb.add(self.tab_motion, text='Motion')
        self._build_motion_tab(self.tab_motion)

        self.tab_seq = ttk.Frame(nb)
        nb.add(self.tab_seq, text='Sequence')
        self._build_seq_tab(self.tab_seq)

        self.tab_prog = ttk.Frame(nb)
        nb.add(self.tab_prog, text='Programs')
        self._build_prog_tab(self.tab_prog)

    def _build_motion_tab(self, parent):
        pad = {'padx': 6, 'pady': 4}

        f1 = ttk.LabelFrame(parent, text='Motion Presets')
        f1.pack(fill='x', **pad)
        row = [('HOME','HOME'), ('READY','READY'), ('PICK','PICK'), ('PLACE','PLACE')]
        fr = ttk.Frame(f1)
        fr.pack(fill='x', padx=4, pady=2)
        for lbl, key in row:
            ttk.Button(fr, text=lbl, width=8,
                       command=lambda k=key: self._go_preset(k)
                       ).pack(side='left', padx=2, expand=True, fill='x')

        f2 = ttk.LabelFrame(parent, text='Jog (End-Effector, hold to repeat)')
        f2.pack(fill='x', **pad)
        sr = ttk.Frame(f2)
        sr.pack(fill='x', padx=4, pady=2)
        ttk.Label(sr, text='Step:').pack(side='left')
        for sz in [0.01, 0.05, 0.1, 0.5]:
            ttk.Radiobutton(sr, text=str(sz),
                            variable=self.step_size, value=sz).pack(side='left', padx=6)
        jr = ttk.Frame(f2)
        jr.pack(fill='x', padx=4, pady=4)
        for lbl, key in [('LEFT','LEFT'),('RIGHT','RIGHT'),('UP','UP'),('DOWN','DOWN')]:
            self._make_hold_button(
                jr, text=lbl, width=10,
                command=lambda k=key: self._jog(k)
            ).pack(side='left', padx=4, expand=True, fill='x')

        f3 = ttk.LabelFrame(parent, text='Gripper')
        f3.pack(fill='x', **pad)
        gf = ttk.Frame(f3)
        gf.pack(fill='x', padx=4, pady=2)
        ttk.Button(gf, text='OPEN',  width=10,
                   command=lambda: self._go_preset('GRIP_OPEN')).pack(
            side='left', padx=2, expand=True, fill='x')
        ttk.Button(gf, text='CLOSE', width=10,
                   command=lambda: self._go_preset('GRIP_CLOSE')).pack(
            side='left', padx=2, expand=True, fill='x')

        f4 = ttk.LabelFrame(parent, text='Manual Joint Control (rad)')
        f4.pack(fill='x', **pad)
        self.slider_widgets = []
        for i, jn in enumerate(JOINT_NAMES):
            row = ttk.Frame(f4)
            row.pack(fill='x', padx=4, pady=2)
            ttk.Label(row, text=jn, width=10).pack(side='left')
            lo, hi = JOINT_LIMITS[i]
            scale = ttk.Scale(row, from_=lo, to=hi,
                              variable=self.slider_vars[i],
                              orient='horizontal')
            scale.pack(side='left', fill='x', expand=True, padx=4)
            self._make_hold_button(
                row, text='-', width=3,
                command=lambda idx=i: self._bump(idx, -1)
            ).pack(side='left', padx=2)
            self._make_hold_button(
                row, text='+', width=3,
                command=lambda idx=i: self._bump(idx, +1)
            ).pack(side='left', padx=2)
            lbl = ttk.Label(row, text='+0.000', width=8, font=('Consolas', 10))
            lbl.pack(side='left', padx=4)
            self.slider_widgets.append((scale, lbl))

        bf = ttk.Frame(parent)
        bf.pack(fill='x', pady=8)
        ttk.Button(bf, text='STOP',         command=self._stop).pack(side='left', padx=4)
        ttk.Button(bf, text='HOME',         command=lambda: self._go_preset('HOME')).pack(side='left', padx=4)
        ttk.Button(bf, text='Sync Sliders', command=self._sync_sliders).pack(side='left', padx=4)
        ttk.Button(bf, text='Quit',         command=self._on_close).pack(side='right', padx=4)

        self.status = tk.StringVar(value='Ready (HOME)')
        ttk.Label(parent, textvariable=self.status,
                  font=('Arial', 10, 'italic')).pack(pady=4)

    def _build_seq_tab(self, parent):
        pad = {'padx': 6, 'pady': 4}

        f1 = ttk.LabelFrame(parent, text='Sequence Steps (click a row to preview)')
        f1.pack(fill='both', expand=True, **pad)

        tf = ttk.Frame(f1)
        tf.pack(fill='both', expand=True, padx=4, pady=2)

        cols = ('idx', 'name', 'j1', 'j2', 'j5', 'j6', 'j7', 'dwell')
        self.tree = ttk.Treeview(tf, columns=cols, show='headings',
                                 height=10, selectmode='browse')
        widths = [40, 160, 70, 70, 70, 70, 70, 70]
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c.upper())
            self.tree.column(c, width=w, anchor='center')
        self.tree.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(tf, orient='vertical', command=self.tree.yview)
        sb.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=sb.set)

        # Row highlight tags (used by _seq_highlight during playback)
        self.tree.tag_configure('active', background='#ffe082', foreground='#000')
        self.tree.tag_configure('done',   foreground='#888')

        # Bindings: single-click row -> preview that step; double-click -> Capture pose.
        # Use <Button-1> (press) + identify_row() so we don't depend on selection-state timing.
        self.tree.bind('<Button-1>', self._on_tree_click, add='+')
        self.tree.bind('<Double-1>', lambda e: self._seq_capture())

        f2 = ttk.LabelFrame(parent, text='Actions')
        f2.pack(fill='x', **pad)

        r1 = ttk.Frame(f2); r1.pack(fill='x', padx=4, pady=2)
        for text, cmd in [
            ('+ Add (Current)',  self._seq_add_current),
            ('+ Add (Preset)',   self._seq_add_preset),
            ('Go to Step',       self._seq_goto_selected),
            ('Capture Pose',     self._seq_capture),
            ('Edit Name/Dwell',  self._seq_edit_meta),
            ('Delete',           self._seq_delete),
            ('Up',               lambda: self._seq_move(-1)),
            ('Down',             lambda: self._seq_move(+1)),
        ]:
            ttk.Button(r1, text=text, command=cmd).pack(side='left', padx=2)

        r2 = ttk.Frame(f2); r2.pack(fill='x', padx=4, pady=2)
        for text, cmd in [
            ('Play',            self._seq_play),
            ('Stop',            self._seq_stop),
            ('Save JSON',       self._seq_save),
            ('Load JSON',       self._seq_load),
            ('Clear All',       self._seq_clear),
        ]:
            ttk.Button(r2, text=text, command=cmd).pack(side='left', padx=2)
        # Loop checkbox — when ticked, Play replays the sequence indefinitely
        # (until Stop or unticked).
        ttk.Checkbutton(
            r2, text='Loop', variable=self.loop_sequence
        ).pack(side='left', padx=10)

        self.seq_status = tk.StringVar(value='No sequence loaded. Add steps to begin.')
        ttk.Label(parent, textvariable=self.seq_status,
                  font=('Arial', 10, 'italic')).pack(pady=4)

    # ----- Motion handlers ------------------------------------------------

    def _go_preset(self, key):
        pose = list(PRESETS[key])
        with self.lock:
            self.last_target = list(self.current)
            self.last_target_time = time.time()
            self.target = pose
            self.current_duration = INTERP_PRESET
        for i, v in enumerate(pose):
            self.slider_vars[i].set(v)
        self.status.set(f'-> {key}')

    def _jog(self, direction):
        mapping = {'LEFT': (JOG_LEFT, -1), 'RIGHT': (JOG_RIGHT, +1),
                   'UP':   (JOG_UP,   -1), 'DOWN':  (JOG_DOWN,  +1)}
        joint_idx, sign = mapping[direction]
        delta = float(self.step_size.get()) * sign
        lo, hi = JOINT_LIMITS[joint_idx]
        with self.lock:
            new_target = list(self.target)
            new_target[joint_idx] = clamp(new_target[joint_idx] + delta, lo, hi)
            self.last_target = list(self.current)
            self.last_target_time = time.time()
            self.target = new_target
            self.current_duration = INTERP_JOG
        self.slider_vars[joint_idx].set(new_target[joint_idx])
        self.status.set(f'{direction}: {JOINT_NAMES[joint_idx]}={new_target[joint_idx]:+.3f}')

    def _bump(self, idx, direction):
        delta = float(self.step_size.get()) * direction
        lo, hi = JOINT_LIMITS[idx]
        with self.lock:
            new_target = list(self.target)
            new_target[idx] = clamp(new_target[idx] + delta, lo, hi)
            self.last_target = list(self.current)
            self.last_target_time = time.time()
            self.target = new_target
            self.current_duration = INTERP_STEP
        self.slider_vars[idx].set(new_target[idx])
        self.status.set(f'Joint {idx+1} {"+" if direction>0 else "-"}{delta:.2f}')

    def _stop(self):
        with self.lock:
            self.last_target = list(self.current)
            self.last_target_time = time.time()
            self.target = list(self.current)
        self.status.set('STOPPED')

    def _sync_sliders(self):
        with self.lock:
            cur = list(self.current)
        for i, v in enumerate(cur):
            self.slider_vars[i].set(v)

    def _on_close(self):
        self.running = False
        self.play_stop.set()
        try:
            self.root.after(150, self.root.destroy)
        except Exception:
            pass

    # ----- Publisher loop -------------------------------------------------

    def _publisher_loop(self):
        """
        Note: Tk DoubleVar.get() requires holding the Tcl interp mutex and
        will block briefly if the main thread is busy (e.g. slider drag).
        We therefore read slider vars OUTSIDE self.lock to avoid holding our
        lock during Tk IPC. The race between slider read and click handler
        is bridged by skip_snap_count set inside _seq_command_to.
        """
        rate = rospy.Rate(PUBLISH_RATE_HZ)
        last_poll = 0.0
        poll_period = 1.0 / POLL_HZ
        while self.running and not rospy.is_shutdown():
            now = time.time()
            # Periodic slider poll (read OUTSIDE lock)
            if now - last_poll >= poll_period:
                last_poll = now
                try:
                    vals = [self.slider_vars[i].get() for i in range(N)]
                except Exception:
                    vals = None
                if vals is not None:
                    with self.lock:
                        # Skip a few polls after _seq_command_to to absorb
                        # the publisher/click-handler race (vals may be stale).
                        if self.skip_snap_count > 0:
                            self.skip_snap_count -= 1
                        elif not self.suppress_slider_snap:
                            if any(abs(vals[i] - self.target[i]) > 1e-4 for i in range(N)):
                                self.target = list(vals)
                                self.last_target = list(vals)
                                self.current = list(vals)
                                self.last_target_time = now - INTERP_PRESET - 1

            with self.lock:
                # Smooth interpolation from last_target -> target
                dur = max(0.01, self.current_duration)
                elapsed = now - self.last_target_time
                t = min(1.0, max(0.0, elapsed / dur))
                e = 3*t*t - 2*t*t*t
                snap = []
                for i in range(N):
                    a = self.last_target[i]
                    b = self.target[i]
                    snap.append(a + (b - a) * e)
                self.current = snap
                pose = list(self.current)
            if pose is not None:
                self._publish(pose)
            rate.sleep()

    def _publish(self, pose):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = list(JOINT_NAMES)
        msg.position = [float(p) for p in pose]
        msg.velocity = [0.0]*N
        msg.effort = [0.0]*N
        try:
            self.js_pub.publish(msg)
        except Exception:
            pass

    # ----- GUI tick (main thread) -----------------------------------------

    def _tick_gui(self):
        """Main-thread tick (~20Hz). All Tk IPC MUST land here so that
        background threads (publisher, seq_run) never call Tk directly."""
        try:
            with self.lock:
                cur = list(self.current)
                pending_slider = self.pending_slider
                self.pending_slider = None
                pending_status = self.pending_status
                self.pending_status = None
            # Update joint-value labels (cheap, safe on main thread)
            for (_, lbl), v in zip(self.slider_widgets, cur):
                lbl.config(text=f'{v:+.3f}')
            # Apply any pending slider values requested by non-main threads
            if pending_slider is not None:
                for i, v in enumerate(pending_slider):
                    self.slider_vars[i].set(v)
            # Apply any pending status string requested by non-main threads
            if pending_status is not None:
                self.seq_status.set(pending_status)
        except Exception:
            pass
        if self.running:
            self.root.after(50, self._tick_gui)

    # ----- Sequence operations --------------------------------------------

    def _seq_refresh(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for i, step in enumerate(self.sequence):
            self.tree.insert('', 'end', values=(
                i + 1,
                step['name'],
                f'{step["pose"][0]:+.3f}',
                f'{step["pose"][1]:+.3f}',
                f'{step["pose"][2]:+.3f}',
                f'{step["pose"][3]:+.3f}',
                f'{step["pose"][4]:+.3f}',
                f'{step["dwell"]:.1f}',
            ))

    def _seq_select_idx(self, idx):
        """Programmatically select row idx (0-based) and scroll into view."""
        rows = self.tree.get_children()
        if 0 <= idx < len(rows):
            self.tree.selection_set(rows[idx])
            self.tree.see(rows[idx])

    def _seq_selected_idx(self):
        sel = self.tree.selection()
        if not sel:
            return -1
        try:
            return int(self.tree.item(sel[0])['values'][0]) - 1
        except Exception:
            return -1

    def _seq_command_to(self, pose):
        """
        Single-shot 'go to pose with smooth interpolation'. Used by both
        single-step playback (Go to Step) and the Play loop.

        IMPORTANT: this is called from BOTH main thread (click handlers) and
        a background thread (_seq_run during Play). We do NOT touch Tk
        widgets directly here (no DoubleVar.set()) — instead we stash the
        new slider values in `pending_slider` and let _tick_gui (which runs
        on main thread) apply them. Stops Tk-interp queue buildup.
        """
        with self.lock:
            if all(abs(self.target[i] - pose[i]) < 1e-3 for i in range(N)):
                # Already targeting this pose — keep slider visually in sync.
                self.pending_slider = list(pose)
                return
            self.last_target = list(self.current)
            self.last_target_time = time.time()
            self.target = list(pose)
            self.current_duration = INTERP_SEQ
            self.pending_slider = list(pose)
            # Suppress the next few snap polls (~300 ms) so a stale
            # slider read taken before the tick_gui tick can apply cannot
            # override our target.
            self.skip_snap_count = 3

    def _seq_add_current(self):
        with self.lock:
            pose = list(self.current)
        self.sequence.append({'name': f'Step{len(self.sequence)+1}',
                              'pose': pose, 'dwell': 1.0})
        self._seq_refresh()
        self._seq_select_idx(len(self.sequence) - 1)
        self.seq_status.set(f'Added current pose as Step{len(self.sequence)}')

    def _seq_add_preset(self):
        dlg = tk.Toplevel(self.root)
        dlg.title('Add Preset Step')
        dlg.geometry('320x200')
        dlg.transient(self.root); dlg.grab_set()

        ttk.Label(dlg, text='Preset:').pack(pady=(10, 2))
        preset_var = tk.StringVar(value='HOME')
        ttk.Combobox(dlg, textvariable=preset_var,
                     values=list(PRESETS.keys()), state='readonly').pack(pady=4)
        ttk.Label(dlg, text='Dwell (s):').pack(pady=(8, 2))
        dwell_var = tk.DoubleVar(value=1.0)
        ttk.Entry(dlg, textvariable=dwell_var).pack(pady=4)

        def ok():
            self.sequence.append({
                'name': preset_var.get(),
                'pose': list(PRESETS[preset_var.get()]),
                'dwell': float(dwell_var.get()),
            })
            self._seq_refresh()
            self._seq_select_idx(len(self.sequence) - 1)
            self.seq_status.set(f'Added preset {preset_var.get()}')
            dlg.destroy()
        ttk.Button(dlg, text='Add', command=ok).pack(pady=10)

    def _on_tree_click(self, event=None):
        """Mouse press on a tree row -> go to that step (preview).
        Uses identify_row() so it works without relying on selection timing."""
        if event is None:
            # Fallback path (e.g. explicit "Go to Step" button)
            idx = self._seq_selected_idx()
            if 0 <= idx < len(self.sequence):
                self._seq_goto(idx)
            else:
                self.seq_status.set('No step selected. Click a step first.')
            return
        row = self.tree.identify_row(event.y)
        if not row:
            return
        try:
            idx = int(self.tree.item(row)['values'][0]) - 1
        except (ValueError, IndexError, KeyError):
            return
        if 0 <= idx < len(self.sequence):
            self._seq_goto(idx)
        else:
            self.seq_status.set(f'Click: idx {idx} out of range (seq len {len(self.sequence)})')

    def _seq_goto_selected(self):
        idx = self._seq_selected_idx()
        if 0 <= idx < len(self.sequence):
            self._seq_goto(idx)
        else:
            messagebox.showinfo('Go to Step', 'Select a step in the table first.')

    def _seq_goto(self, idx):
        """Move the robot smoothly to a single step."""
        if not (0 <= idx < len(self.sequence)):
            return
        step = self.sequence[idx]
        self._seq_command_to(step['pose'])
        self.seq_status.set(f'-> step {idx+1}/{len(self.sequence)} ({step["name"]})')
        # Highlight this row
        self._seq_highlight_async(idx)

    # --- Row highlight (thread-safe via root.after) ---

    def _seq_highlight_async(self, idx):
        """Schedule a highlight update on the main thread."""
        self.root.after(0, lambda: self._seq_highlight(idx))

    def _seq_clear_highlight_async(self):
        self.root.after(0, self._seq_clear_highlight)

    def _seq_highlight(self, idx):
        """Mark row idx as active, all earlier as done, all later untagged. (Main thread only.)"""
        rows = self.tree.get_children()
        for i, row in enumerate(rows):
            if i == idx:
                self.tree.item(row, tags=('active',))
            elif i < idx:
                self.tree.item(row, tags=('done',))
            else:
                self.tree.item(row, tags=())

    def _seq_clear_highlight(self):
        rows = self.tree.get_children()
        for row in rows:
            self.tree.item(row, tags=())

    # ----- Programs tab ----------------------------------------------------

    def _build_prog_tab(self, parent):
        """Third tab: load saved JSON sequences and run them as a master."""
        pad = {'padx': 6, 'pady': 4}

        f1 = ttk.LabelFrame(parent, text='Loaded Programs')
        f1.pack(fill='both', expand=True, **pad)

        tf = ttk.Frame(f1)
        tf.pack(fill='both', expand=True, padx=4, pady=2)

        cols = ('idx', 'name', 'steps', 'file')
        self.prog_tree = ttk.Treeview(tf, columns=cols, show='headings',
                                      height=10, selectmode='browse')
        widths = [40, 280, 70, 240]
        for c, w in zip(cols, widths):
            self.prog_tree.heading(c, text=c.upper())
            self.prog_tree.column(c, width=w, anchor='center')
        # Hide the FILE column by default but show as a tooltip-ish via state
        self.prog_tree.column('file', width=0, stretch=False)
        self.prog_tree.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(tf, orient='vertical', command=self.prog_tree.yview)
        sb.pack(side='right', fill='y')
        self.prog_tree.configure(yscrollcommand=sb.set)

        # Row highlight for active program during Play
        self.prog_tree.tag_configure('active', background='#a5d6a7', foreground='#000')
        self.prog_tree.tag_configure('done',   foreground='#888')

        f2 = ttk.LabelFrame(parent, text='Actions')
        f2.pack(fill='x', **pad)

        r1 = ttk.Frame(f2); r1.pack(fill='x', padx=4, pady=2)
        for text, cmd in [
            ('+ Load Sequence', self._prog_load),
            ('Remove',          self._prog_remove),
            ('Up',              lambda: self._prog_move(-1)),
            ('Down',            lambda: self._prog_move(+1)),
            ('Clear All',       self._prog_clear),
        ]:
            ttk.Button(r1, text=text, command=cmd).pack(side='left', padx=2)

        r2 = ttk.Frame(f2); r2.pack(fill='x', padx=4, pady=2)
        for text, cmd in [
            ('Play All', self._prog_play),
            ('Stop',     self._prog_stop),
        ]:
            ttk.Button(r2, text=text, command=cmd).pack(side='left', padx=2)
        ttk.Checkbutton(
            r2, text='Loop', variable=self.loop_programs
        ).pack(side='left', padx=10)

        self.prog_status = tk.StringVar(
            value='No programs loaded. Click "+ Load Sequence" to add JSON files.')
        ttk.Label(parent, textvariable=self.prog_status,
                  font=('Arial', 10, 'italic')).pack(pady=4)

    def _prog_refresh(self):
        for row in self.prog_tree.get_children():
            self.prog_tree.delete(row)
        for i, p in enumerate(self.programs):
            self.prog_tree.insert('', 'end', values=(
                i + 1, p['name'], len(p['steps']), p['path']))

    def _prog_selected_idx(self):
        sel = self.prog_tree.selection()
        if not sel:
            return -1
        try:
            return int(self.prog_tree.item(sel[0])['values'][0]) - 1
        except Exception:
            return -1

    def _prog_load(self):
        default_dir = os.path.expanduser('~/dobot_sequences')
        os.makedirs(default_dir, exist_ok=True)
        fp = filedialog.askopenfilename(
            title='Load Sequence JSON',
            filetypes=[('JSON', '*.json')],
            initialdir=default_dir)
        if not fp:
            return
        try:
            with open(fp) as f:
                data = json.load(f)
            steps = data.get('steps', [])
            if not steps:
                messagebox.showinfo('Load', 'No steps in this sequence.')
                return
            self.programs.append({
                'name': os.path.basename(fp),
                'path': fp,
                'steps': steps,
            })
            self._prog_refresh()
            self.prog_status.set(
                f'Loaded {os.path.basename(fp)} ({len(steps)} steps). '
                f'{len(self.programs)} programs total.')
        except Exception as e:
            messagebox.showerror('Load error', str(e))

    def _prog_remove(self):
        idx = self._prog_selected_idx()
        if idx < 0:
            return
        name = self.programs[idx]['name']
        del self.programs[idx]
        self._prog_refresh()
        new_sel = min(idx, len(self.programs) - 1)
        if new_sel >= 0:
            iid = self.prog_tree.get_children()[new_sel]
            self.prog_tree.selection_set(iid)
        self.prog_status.set(f'Removed "{name}".')

    def _prog_move(self, delta):
        idx = self._prog_selected_idx()
        if idx < 0:
            return
        j = idx + delta
        if 0 <= j < len(self.programs):
            self.programs[idx], self.programs[j] = self.programs[j], self.programs[idx]
            self._prog_refresh()
            iid = self.prog_tree.get_children()[j]
            self.prog_tree.selection_set(iid)

    def _prog_clear(self):
        if not self.programs:
            return
        if messagebox.askyesno('Clear', f'Clear all {len(self.programs)} programs?'):
            self.programs = []
            self._prog_refresh()
            self.prog_status.set('Cleared.')

    def _prog_highlight_async(self, idx):
        self.root.after(0, lambda: self._prog_highlight(idx))

    def _prog_clear_highlight_async(self):
        self.root.after(0, self._prog_clear_highlight)

    def _prog_highlight(self, idx):
        rows = self.prog_tree.get_children()
        for i, row in enumerate(rows):
            if i == idx:
                self.prog_tree.item(row, tags=('active',))
            elif i < idx:
                self.prog_tree.item(row, tags=('done',))
            else:
                self.prog_tree.item(row, tags=())

    def _prog_clear_highlight(self):
        for row in self.prog_tree.get_children():
            self.prog_tree.item(row, tags=())

    def _prog_play(self):
        if self.prog_playing:
            self.prog_status.set('Already playing.')
            return
        if not self.programs:
            messagebox.showinfo('Play', 'No programs loaded.')
            return
        self.prog_stop.clear()
        self.prog_playing = True
        # Block slider snap during program playback too
        self.suppress_slider_snap = True
        threading.Thread(target=self._prog_run, daemon=True).start()

    def _prog_stop(self):
        self.prog_stop.set()
        self.prog_status.set('Stop requested...')

    def _prog_run(self):
        """Background thread: play every loaded program in order (optionally looped)."""
        prog_replays = 0
        try:
            while True:
                for i, prog in enumerate(self.programs):
                    if self.prog_stop.is_set():
                        break
                    steps = prog.get('steps', [])
                    with self.lock:
                        self.pending_status = (
                            f'[{i+1}/{len(self.programs)}] {prog["name"]} '
                            f'({len(steps)} steps)'
                            + (f' (loop {prog_replays})' if prog_replays > 0 else ''))
                    self._prog_highlight_async(i)
                    for step in steps:
                        if self.prog_stop.is_set():
                            break
                        pose = step.get('pose', list(PRESETS['HOME']))
                        dwell = float(step.get('dwell', 1.0))
                        name = step.get('name', '?')
                        with self.lock:
                            self.pending_status = (
                                f'  {prog["name"]} [{step.get("name","?")}]'
                                f' (dwell {dwell:.1f}s)')
                        self._seq_command_to(pose)
                        total = INTERP_SEQ + dwell
                        t0 = time.time()
                        while time.time() - t0 < total:
                            if self.prog_stop.is_set():
                                break
                            time.sleep(0.05)
                if self.prog_stop.is_set():
                    break
                if not self.loop_programs.get():
                    break
                self._prog_clear_highlight_async()
                prog_replays += 1
                time.sleep(0.3)
        finally:
            self.suppress_slider_snap = False
            self.prog_playing = False
            with self.lock:
                if self.prog_stop.is_set():
                    self.pending_status = 'Programs stopped.'
                elif prog_replays > 0:
                    self.pending_status = f'Programs done (looped {prog_replays}\u00d7).'
                else:
                    self.pending_status = 'Programs done.'
            # Hold highlight on the last program briefly, then clear
            if self.programs:
                self._prog_highlight_async(len(self.programs) - 1)
            self.root.after(1500, self._prog_clear_highlight_async)

    def _seq_capture(self):
        idx = self._seq_selected_idx()
        if idx < 0:
            messagebox.showinfo('Capture', 'Select a step first.')
            return
        with self.lock:
            pose = list(self.current)
        self.sequence[idx]['pose'] = pose
        self._seq_refresh()
        self._seq_select_idx(idx)
        self.seq_status.set(f'Captured current pose to step {idx+1}')

    def _seq_edit_meta(self):
        idx = self._seq_selected_idx()
        if idx < 0:
            messagebox.showinfo('Edit', 'Select a step first.')
            return
        s = self.sequence[idx]
        dlg = tk.Toplevel(self.root)
        dlg.title(f'Edit Step {idx+1}')
        dlg.geometry('300x150')
        dlg.transient(self.root); dlg.grab_set()
        ttk.Label(dlg, text='Name:').grid(row=0, column=0, sticky='w', padx=8, pady=8)
        name_var = tk.StringVar(value=s['name'])
        ttk.Entry(dlg, textvariable=name_var).grid(row=0, column=1, padx=8, pady=8)
        ttk.Label(dlg, text='Dwell (s):').grid(row=1, column=0, sticky='w', padx=8, pady=8)
        dwell_var = tk.DoubleVar(value=s['dwell'])
        ttk.Entry(dlg, textvariable=dwell_var).grid(row=1, column=1, padx=8, pady=8)
        def ok():
            self.sequence[idx]['name'] = name_var.get()
            self.sequence[idx]['dwell'] = float(dwell_var.get())
            self._seq_refresh()
            self._seq_select_idx(idx)
            dlg.destroy()
        ttk.Button(dlg, text='OK', command=ok).grid(row=2, column=1, pady=12)

    def _seq_delete(self):
        idx = self._seq_selected_idx()
        if idx < 0:
            return
        del self.sequence[idx]
        self._seq_refresh()
        new_sel = min(idx, len(self.sequence) - 1)
        if new_sel >= 0:
            self._seq_select_idx(new_sel)
        self.seq_status.set(f'Deleted step {idx+1}')

    def _seq_move(self, delta):
        idx = self._seq_selected_idx()
        if idx < 0:
            return
        j = idx + delta
        if 0 <= j < len(self.sequence):
            self.sequence[idx], self.sequence[j] = self.sequence[j], self.sequence[idx]
            self._seq_refresh()
            self._seq_select_idx(j)

    def _seq_play(self):
        if self.playing:
            self.seq_status.set('Already playing.')
            return
        if not self.sequence:
            messagebox.showinfo('Play', 'Sequence is empty.')
            return
        # If user clicks Play mid-interpolation, snap current to slider values
        # so sequence starts from the currently visible robot state.
        with self.lock:
            cur = list(self.current)
        for i, v in enumerate(cur):
            self.slider_vars[i].set(v)
        # Block any slider-snap in the publisher for the entire Play so stale
        # slider reads (Tk IPC from non-main thread can lag) cannot override
        # the step targets. Re-enabled when _seq_run finishes or is stopped.
        self.suppress_slider_snap = True
        self.play_stop.clear()
        self.playing = True
        threading.Thread(target=self._seq_run, daemon=True).start()

    def _seq_stop(self):
        self.play_stop.set()
        self.seq_status.set('Stop requested...')

    def _seq_run(self):
        replay_count = 0  # how many times the whole sequence has finished
        try:
            while True:
                for i, step in enumerate(self.sequence):
                    if self.play_stop.is_set():
                        break
                    # Use pending_status so we don't call Tk from this background
                    # thread. _tick_gui will apply the string on main thread.
                    suffix = (f' (replay {replay_count})'
                              if replay_count > 0 else '')
                    with self.lock:
                        self.pending_status = (
                            f'[{i+1}/{len(self.sequence)}] -> {step["name"]} '
                            f'(dwell {step["dwell"]:.1f}s){suffix}')
                    self._seq_highlight_async(i)         # highlight current row
                    self._seq_command_to(step['pose'])
                    total = INTERP_SEQ + step['dwell']
                    t0 = time.time()
                    while time.time() - t0 < total:
                        if self.play_stop.is_set():
                            break
                        time.sleep(0.05)
                # End of one pass — decide whether to stop, loop, or break
                if self.play_stop.is_set():
                    break
                if not self.loop_sequence.get():
                    break
                # Loop is on: clear highlight before replaying so the user
                # sees the rows flash off and back on at the start of each loop.
                self._seq_clear_highlight_async()
                replay_count += 1
                # Brief pause between loops so the transition is visible
                time.sleep(0.3)
        finally:
            # Always re-enable slider snap — even on exception or early Stop —
            # so the Motion-tab sliders work again after Play ends.
            self.suppress_slider_snap = False
            self.playing = False
            with self.lock:
                if self.play_stop.is_set():
                    self.pending_status = 'Sequence stopped.'
                elif replay_count > 0:
                    self.pending_status = f'Sequence done (looped {replay_count} \u00d7).'
                else:
                    self.pending_status = 'Sequence done.'
            # Highlight the very last step (so user can see where it ended),
            # then strip all tags after a short pause.
            if self.sequence:
                self._seq_highlight_async(len(self.sequence) - 1)
            self.root.after(1500, self._seq_clear_highlight)

    def _seq_save(self):
        if not self.sequence:
            messagebox.showinfo('Save', 'No steps to save.')
            return
        default_dir = os.path.expanduser('~/dobot_sequences')
        os.makedirs(default_dir, exist_ok=True)
        fp = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[('JSON','*.json')],
            initialdir=default_dir,
            initialfile='sequence.json')
        if not fp:
            return
        data = {'version': 1,
                'joint_names': JOINT_NAMES,
                'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                'steps': self.sequence}
        try:
            with open(fp, 'w') as f:
                json.dump(data, f, indent=2)
            self.seq_status.set(f'Saved {len(self.sequence)} steps -> {fp}')
        except Exception as e:
            messagebox.showerror('Save error', str(e))

    def _seq_load(self):
        default_dir = os.path.expanduser('~/dobot_sequences')
        os.makedirs(default_dir, exist_ok=True)
        fp = filedialog.askopenfilename(
            filetypes=[('JSON','*.json')],
            initialdir=default_dir)
        if not fp:
            return
        try:
            with open(fp) as f:
                data = json.load(f)
            self.sequence = data['steps']
            self._seq_refresh()
            if self.sequence:
                self._seq_select_idx(0)
            self.seq_status.set(f'Loaded {len(self.sequence)} steps from {fp}')
        except Exception as e:
            messagebox.showerror('Load error', str(e))

    def _seq_clear(self):
        if not self.sequence:
            return
        if messagebox.askyesno('Clear', f'Clear all {len(self.sequence)} steps?'):
            self.sequence = []
            self._seq_refresh()
            self.seq_status.set('Cleared.')


if __name__ == '__main__':
    try:
        root = tk.Tk()
        app = DobotGUI(root)
        root.mainloop()
    except rospy.ROSException:
        print('ERROR: ROS master not running. Start it with:  roscore')
    except KeyboardInterrupt:
        pass
    finally:
        try:
            app._on_close()
        except Exception:
            pass
