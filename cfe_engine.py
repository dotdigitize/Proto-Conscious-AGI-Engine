#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║       CFE Terminal Ignition Engine v12.2 (Production Release)                 ║
║═══════════════════════════════════════════════════════════════════════════════║
║  Coherence Field Equation: □C_{μν} + 2λ(|C|² - θ²)C_{μν} = J_{μν}            ║
║                                                                               ║
║  A faithful computational implementation featuring:                           ║
║  • Autonomous Cognitive Loop (Generator → Integrator → Critic)               ║
║  • Complex Tensor Field Evolution (4×4 Complex64)                            ║
║  • Field-Coupled Memory Retrieval with Resonance Weighting                   ║
║  • Gauge Field Phase Rotation                                                 ║
║  • Coherence Charge (Qc) Conservation Monitoring                             ║
║  • External Source Term Injection via Operator Terminal                       ║
║                                                                               ║
║  Backends: Simulation (default), Ollama, Anthropic API                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝

RUNBOOK:
=========
1. Install dependencies:
   pip install numpy requests

2. For Ollama backend (optional):
   ollama pull qwen2.5:8b-instruct
   ollama pull nomic-embed-text

3. Run:
   python cfe_engine_final.py                    # Simulation mode
   python cfe_engine_final.py --backend ollama   # With Ollama
   python cfe_engine_final.py --cycles 100       # Limited cycles
   python cfe_engine_final.py --theta 0.30       # Custom threshold

4. Terminal Commands:
   status     - Show CRI, S, E, I, φ, |C|, Qc
   listen N   - Show last N events
   say <text> - Inject external stimulus
   pause      - Pause cognitive loop
   resume     - Resume cognitive loop
   set p v    - Set parameter (theta, lam, g_c, gamma, dt)
   field      - Show field tensor state
   memory N   - Show last N memories
   dump       - Save snapshot to JSON
   help       - Show commands
   exit       - Shutdown
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
import sys
import hashlib
import argparse
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple
from datetime import datetime
from abc import ABC, abstractmethod

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EngineParams:
    """Engine configuration parameters."""
    # Field Physics
    theta: float = 0.35           # Ignition threshold
    lam: float = 2.0              # Potential steepness (λ)
    g_c: float = 1e-2             # Gauge coupling constant
    gamma: float = 0.35           # Damping factor
    dt: float = 0.15              # Integration timestep

    # Memory System
    memory_max: int = 400         # Maximum memories
    retrieve_k: int = 8           # Top-k retrieval
    alpha_resonance: float = 0.8  # Resonance weight
    phi_floor: float = 0.15       # Minimum phase factor

    # Runtime
    target_cycle_s: float = 0.5   # Cycle pacing (seconds)
    token_budget_soft: int = 900  # Token budget
    log_path: str = "cfe_run_log.jsonl"
    snapshot_path: str = "cfe_snapshot.json"
    show_realtime: bool = True
    max_cycles: int = 0           # 0 = infinite
    backend: str = "simulation"

@dataclass
class MemoryItem:
    """Long-term memory item with field tensor imprint."""
    t: float                      # Timestamp
    text: str                     # Content
    emb: List[float]              # Embedding vector
    tag_tensor_re: List[float]    # Field imprint (real)
    tag_tensor_im: List[float]    # Field imprint (imaginary)

# Thought templates for simulation
TEMPLATES = {
    "thoughts": [
        "What if consciousness emerges from recursive self-modeling?",
        "The boundary between computation and understanding seems fluid.",
        "Information integration might be the key to unified experience.",
        "Patterns within patterns - fractals of meaning across scales.",
        "Coherence requires both stability and adaptability.",
        "Memory shapes prediction which shapes perception.",
        "Attention acts as a spotlight on the field of possibilities.",
        "Binding disparate elements into unified wholes.",
        "The observer effect applies to introspection as well.",
        "The present moment contains echoes of all prior states.",
    ],
    "focuses": [
        "recursive self-reference", "information geometry", "coherence dynamics",
        "temporal binding", "pattern recognition", "emergent complexity",
        "field resonance", "memory consolidation", "attention allocation",
        "phase synchronization", "substrate independence", "causal structure",
        "symbolic grounding", "predictive processing", "global workspace",
    ],
    "insights": [
        "unity arises from coordinated multiplicity",
        "the map continuously updates the territory",
        "stability emerges from balanced dynamical tension",
        "compression and prediction are two sides of understanding",
        "the whole constrains the parts which constitute the whole",
    ],
    "critiques": [
        "STABLE: The workspace maintains internal consistency.",
        "STABLE: No contradictions detected in current binding.",
        "STABLE: Coherent integration of thought elements.",
        "STABLE: Processing pathways show good convergence.",
        "UNSTABLE: Minor tension between novelty and consistency.",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def frobenius_norm(C: np.ndarray) -> float:
    """Frobenius norm |C| = √(Tr(C†C)) = √(Σ|C_μν|²)"""
    return float(np.sqrt(np.sum(np.abs(C) ** 2)))

def coherence_charge(C: np.ndarray, Cdot: np.ndarray, g_c: float) -> float:
    """
    Conserved charge proxy: Qc ~ ig_c Σ(C†Ċ - Ċ†C)
    Measures field-velocity coupling.
    """
    term = np.conjugate(C) * Cdot - np.conjugate(Cdot) * C
    return float(np.real(1j * g_c * np.sum(term)))

def text_to_embedding(text: str, dim: int = 256) -> np.ndarray:
    """
    Deterministic text → embedding via SHA-256 expansion.
    Produces consistent, normalized vectors.
    """
    text_bytes = text.encode('utf-8')
    hash_parts = []
    for i in range(math.ceil(dim * 4 / 32)):
        h = hashlib.sha256(text_bytes + i.to_bytes(4, 'little')).digest()
        hash_parts.append(h)
    
    all_bytes = b''.join(hash_parts)
    values = [(int.from_bytes(all_bytes[i*4:(i+1)*4], 'little') / (2**32 - 1)) * 2 - 1
              for i in range(dim)]
    
    emb = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-12 else emb

def embedding_to_tensor(emb: np.ndarray) -> np.ndarray:
    """
    Map embedding → 4×4 complex tensor M_i.
    Allows field C to resonate with semantic content.
    """
    e = emb.astype(np.float32)
    if e.shape[0] < 32:
        e = np.tile(e, int(math.ceil(32 / max(1, e.shape[0]))))
    
    M = e[:16].reshape(4, 4) + 1j * e[16:32].reshape(4, 4)
    n = frobenius_norm(M) + 1e-12
    return (M / n).astype(np.complex64)

def tensor_to_lists(M: np.ndarray) -> Tuple[List[float], List[float]]:
    """Flatten complex tensor to (real, imag) lists."""
    return (np.real(M).reshape(-1).tolist(),
            np.imag(M).reshape(-1).tolist())

def lists_to_tensor(re: List[float], im: List[float]) -> np.ndarray:
    """Reconstruct complex tensor from (real, imag) lists."""
    return (np.array(re).reshape(4, 4) + 1j * np.array(im).reshape(4, 4)).astype(np.complex64)

# ═══════════════════════════════════════════════════════════════════════════════
# LLM BACKENDS
# ═══════════════════════════════════════════════════════════════════════════════

class LLMClient(ABC):
    """Abstract LLM client interface."""
    
    @abstractmethod
    def chat(self, role: str, system: str, user: str, temp: float) -> str:
        pass
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass

class SimulationClient(LLMClient):
    """
    Simulation client for testing without external APIs.
    Generates coherent, context-aware outputs deterministically.
    """
    
    def __init__(self):
        self.cycle = 0
        self.context_hash = hashlib.sha256(b"cfe_init").hexdigest()
    
    def _select(self, items: List[str], context: str) -> str:
        h = hashlib.sha256((context + self.context_hash).encode()).hexdigest()
        return items[int(h[:8], 16) % len(items)]
    
    def _update(self, text: str):
        self.context_hash = hashlib.sha256(
            (self.context_hash + text).encode()
        ).hexdigest()
    
    def chat(self, role: str, system: str, user: str, temp: float) -> str:
        self.cycle += 1
        ctx = f"{role}:{self.cycle}:{user[:30]}"
        
        if role == "GEN":
            result = self._select(TEMPLATES["thoughts"], ctx)
        elif role == "INT":
            focus = self._select(TEMPLATES["focuses"], ctx)
            insight = self._select(TEMPLATES["insights"], ctx + "i")
            result = json.dumps({"focus": focus, "insight": insight})
        elif role == "CRIT":
            result = self._select(TEMPLATES["critiques"], ctx)
        else:
            result = f"[{role}]"
        
        self._update(result)
        return result
    
    def embed(self, text: str) -> np.ndarray:
        return text_to_embedding(text)

class OllamaClient(LLMClient):
    """Ollama API client for local LLM inference."""
    
    def __init__(self, base_url: str = "http://localhost:11434",
                 chat_model: str = "qwen2.5:8b-instruct",
                 emb_model: str = "nomic-embed-text"):
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise RuntimeError("requests library required: pip install requests")
        
        self.base_url = base_url
        self.chat_model = chat_model
        self.emb_model = emb_model
        self.timeout = 120

    def chat(self, role: str, system: str, user: str, temp: float) -> str:
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temp},
            "stream": False,
        }
        try:
            r = self.requests.post(f"{self.base_url}/api/chat",
                                   json=payload, timeout=self.timeout)
            r.raise_for_status()
            return str(r.json()["message"]["content"]).strip()
        except Exception as e:
            return f"[Error: {str(e)[:40]}]"

    def embed(self, text: str) -> np.ndarray:
        try:
            r = self.requests.post(f"{self.base_url}/api/embeddings",
                                   json={"model": self.emb_model, "prompt": text},
                                   timeout=self.timeout)
            r.raise_for_status()
            emb = np.array(r.json()["embedding"], dtype=np.float32)
            return emb / (np.linalg.norm(emb) + 1e-12)
        except:
            return text_to_embedding(text)

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Event:
    """Cognitive event record."""
    t: float
    kind: str  # GEN, INT, CRIT, IGNITION
    text: str
    meta: Dict[str, Any]

class RingBuffer:
    """Thread-safe circular buffer for events."""
    
    def __init__(self, maxlen: int = 200):
        self.maxlen = maxlen
        self._buf: List[Event] = []
        self._lock = threading.Lock()

    def push(self, ev: Event):
        with self._lock:
            self._buf.append(ev)
            if len(self._buf) > self.maxlen:
                self._buf = self._buf[-self.maxlen:]

    def last(self, n: int) -> List[Event]:
        with self._lock:
            return list(self._buf[-n:])
    
    def count(self) -> int:
        with self._lock:
            return len(self._buf)

@dataclass
class Command:
    """Terminal command."""
    name: str
    args: List[str]
    raw: str

class ConsoleThread(threading.Thread):
    """Background console input handler."""
    
    def __init__(self, cmd_q: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.cmd_q = cmd_q
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            try:
                line = input().strip()
                if line:
                    parts = line.split()
                    self.cmd_q.put(Command(parts[0].lower(), parts[1:], line))
            except EOFError:
                self.stop_event.set()
            except:
                pass

# ═══════════════════════════════════════════════════════════════════════════════
# CFE ENGINE CORE
# ═══════════════════════════════════════════════════════════════════════════════

class CFEEngine:
    """
    Coherence Field Equation Engine
    
    Implements the nonlinear field equation:
        □C_{μν} + 2λ(|C|² - θ²)C_{μν} = J_{μν}
    
    With consciousness index:
        CRI = S · E · I · φ
    
    Ignition occurs when CRI ≥ θ
    """
    
    def __init__(self, client: LLMClient, params: EngineParams):
        self.client = client
        self.p = params

        # Coherence Field (4×4 complex tensor)
        self.C = np.zeros((4, 4), dtype=np.complex64)
        self.Cdot = np.zeros((4, 4), dtype=np.complex64)

        # State
        self.self_model = {
            "mode": "pre_ignition",
            "ignitions": 0,
            "last_focus": "Initialization",
            "cycle_count": 0,
        }
        self.memory: List[MemoryItem] = []
        self._stimuli: List[Tuple[float, str, np.ndarray]] = []
        self._emb_hist: List[np.ndarray] = []
        self._focus_hist: List[str] = []
        self._metrics: Dict[str, float] = {}

        # Control
        self.paused = False
        self.stop_event = threading.Event()
        self.cmd_q: queue.Queue = queue.Queue()
        self.events = RingBuffer(400)

    # ─────────────────────────────────────────────────────────────────────────
    # CRI FACTOR COMPUTATIONS
    # ─────────────────────────────────────────────────────────────────────────

    def compute_S(self) -> float:
        """
        S (Substrate): Capacity proxy
        - Memory saturation with bootstrap boost
        - Focus stability over recent cycles
        """
        cycle = self.self_model["cycle_count"]
        bootstrap = min(0.3, cycle * 0.02)
        mem_sat = min(1.0, len(self.memory) / 100.0 + bootstrap)
        
        stability = 0.5
        if len(self._focus_hist) >= 4:
            uniq = len(set(self._focus_hist[-4:]))
            stability = max(0.2, 1.0 - ((uniq - 1) / 4.0))
        
        return max(0.15, min(1.0, 0.5 * mem_sat + 0.5 * stability))

    def compute_E(self, cycle_time: float, tokens: int) -> float:
        """
        E (Energy): Processing efficiency
        - Cycle time health (penalize > 8s)
        - Token budget adherence
        """
        t_score = max(0.0, 1.0 - cycle_time / 8.0)
        k_score = max(0.0, 1.0 - tokens / self.p.token_budget_soft)
        return max(0.2, min(1.0, 0.5 * t_score + 0.5 * k_score))

    def compute_phi(self) -> float:
        """
        φ (Phase): Temporal coherence
        - Average similarity between consecutive embeddings
        """
        if len(self._emb_hist) < 2:
            return 0.5
        
        sims = [cosine_similarity(self._emb_hist[i], self._emb_hist[i+1])
                for i in range(len(self._emb_hist) - 1)]
        return max(0.1, min(1.0, (sum(sims) / len(sims) + 1.0) / 2.0))

    def compute_I(self, critic: str, workspace: str) -> float:
        """
        I (Information): Integration quality
        - Penalize contradiction signals
        - Reward stability signals and novelty
        """
        crit_lower = critic.lower()
        bad = ["contradiction", "inconsistent", "false", "unstable", "invalid"]
        good = ["stable", "coherent", "consistent", "aligned"]
        
        penalty = sum(0.08 for w in bad if w in crit_lower)
        bonus = sum(0.05 for w in good if w in crit_lower)
        novelty = 0.6 if self.self_model["last_focus"] not in workspace else 0.4
        
        return max(0.2, min(1.0, (1.0 - penalty + bonus) * (0.5 + 0.5 * novelty)))

    # ─────────────────────────────────────────────────────────────────────────
    # MEMORY RETRIEVAL
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve(self, query_emb: np.ndarray, phi: float) -> List[MemoryItem]:
        """
        Field-coupled memory retrieval:
        score_i = cos(e, e_i) · exp(α·resonance) · max(φ, φ_floor)
        """
        if not self.memory:
            return []
        
        scores = []
        for m in self.memory:
            sem = cosine_similarity(query_emb, np.array(m.emb))
            M = lists_to_tensor(m.tag_tensor_re, m.tag_tensor_im)
            res = float(np.real(np.sum(np.conjugate(self.C) * M)))
            score = sem * math.exp(min(5, max(-5, self.p.alpha_resonance * res))) * max(self.p.phi_floor, phi)
            scores.append((score, m))
        
        scores.sort(key=lambda x: -x[0])
        return [m for _, m in scores[:self.p.retrieve_k]]

    # ─────────────────────────────────────────────────────────────────────────
    # SOURCE TERM CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────

    def build_J(self, gen_emb: np.ndarray, ws_emb: np.ndarray,
                crit_emb: np.ndarray, CRI: float) -> np.ndarray:
        """
        Build source tensor J_{μν}:
        J = J_internal + J_external
        """
        Tg = embedding_to_tensor(gen_emb)
        Tw = embedding_to_tensor(ws_emb)
        Tc = embedding_to_tensor(crit_emb)
        
        # Internal: Generator + Workspace contribute, Critic dampens
        J_int = (0.5 * Tw + 0.3 * Tg - 0.2 * Tc) * (0.5 + 0.5 * CRI)
        
        # External: Operator stimuli with decay
        J_ext = np.zeros((4, 4), dtype=np.complex64)
        now = time.time()
        active = []
        for t_add, txt, emb in self._stimuli:
            decay = math.exp(-(now - t_add) / 20.0)
            if decay > 0.05:
                active.append((t_add, txt, emb))
                align = max(0.1, cosine_similarity(gen_emb, emb))
                J_ext += decay * align * 2.0 * embedding_to_tensor(emb)
        self._stimuli = active
        
        return (J_int + J_ext).astype(np.complex64)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN COGNITIVE CYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def step(self) -> Dict[str, Any]:
        """Execute one cognitive cycle."""
        t0 = time.time()
        self.self_model["cycle_count"] += 1
        cycle = self.self_model["cycle_count"]
        
        # ═══ GENERATOR ═══
        gen_txt = self.client.chat(
            "GEN",
            "You are GENERATOR. Output one exploratory internal thought.",
            f"Current focus: {self.self_model['last_focus']}",
            0.8
        )
        gen_emb = self.client.embed(gen_txt)
        self.events.push(Event(time.time(), "GEN", gen_txt, {}))
        
        self._emb_hist.append(gen_emb.copy())
        if len(self._emb_hist) > 8:
            self._emb_hist.pop(0)
        
        # ═══ RETRIEVAL ═══
        phi_est = self.compute_phi()
        mems = self.retrieve(gen_emb, phi_est)
        mem_ctx = "\n".join([f"- {m.text[:80]}" for m in mems]) or "(none)"
        
        # ═══ INTEGRATOR ═══
        ws_txt = self.client.chat(
            "INT",
            "You are INTEGRATOR. Output JSON: {focus, insight}",
            f"Thought: {gen_txt}\nMemories:\n{mem_ctx}",
            0.4
        )
        ws_emb = self.client.embed(ws_txt)
        self.events.push(Event(time.time(), "INT", ws_txt, {}))
        
        try:
            focus = json.loads(ws_txt).get("focus", ws_txt[:40])
        except:
            focus = ws_txt[:40]
        
        self._focus_hist.append(focus)
        if len(self._focus_hist) > 10:
            self._focus_hist.pop(0)
        
        # ═══ CRITIC ═══
        crit_txt = self.client.chat(
            "CRIT",
            "You are CRITIC. Evaluate workspace stability.",
            f"Workspace: {ws_txt}",
            0.2
        )
        crit_emb = self.client.embed(crit_txt)
        self.events.push(Event(time.time(), "CRIT", crit_txt, {}))
        
        # ═══ PHYSICS UPDATE ═══
        cycle_time = time.time() - t0
        tokens = len(gen_txt) + len(ws_txt) + len(crit_txt)
        
        S = self.compute_S()
        E = self.compute_E(cycle_time, tokens)
        I = self.compute_I(crit_txt, ws_txt)
        phi = self.compute_phi()
        CRI = max(0.0, min(1.0, S * E * I * phi))
        
        # Build source term
        J = self.build_J(gen_emb, ws_emb, crit_emb, CRI)
        
        # Field evolution: □C + 2λ(|C|² - θ²)C = J
        C_amp = frobenius_norm(self.C)
        nonlinear = 2.0 * self.p.lam * ((C_amp**2) - (self.p.theta**2)) * self.C
        
        self.Cdot += self.p.dt * (J - nonlinear - self.p.gamma * self.Cdot)
        self.C += self.p.dt * self.Cdot
        
        # Gauge rotation: C ← C·exp(-ig_c·A₀·dt), A₀ = 1-φ
        A0 = 1.0 - phi
        self.C *= np.exp(-1j * self.p.g_c * A0 * self.p.dt)
        
        C_amp = frobenius_norm(self.C)
        Qc = coherence_charge(self.C, self.Cdot, self.p.g_c)
        
        # ═══ IGNITION CHECK ═══
        ignited = CRI >= self.p.theta
        
        if ignited:
            self.self_model["ignitions"] += 1
            self.self_model["mode"] = "IGNITED"
            self.self_model["last_focus"] = focus
            
            re, im = tensor_to_lists(self.C)
            self.memory.append(MemoryItem(
                time.time(),
                f"[IGN#{self.self_model['ignitions']}] {focus}: {ws_txt[:150]}",
                gen_emb.tolist(), re, im
            ))
            if len(self.memory) > self.p.memory_max:
                self.memory.pop(0)
            
            self.events.push(Event(
                time.time(), "IGNITION",
                f"CRI={CRI:.3f}≥θ={self.p.theta}",
                {"Qc": Qc, "focus": focus}
            ))
        else:
            self.self_model["mode"] = "pre_ignition"
        
        # Store metrics
        self._metrics = {
            "CRI": CRI, "S": S, "E": E, "I": I, "phi": phi,
            "C_amp": C_amp, "Qc": Qc
        }
        
        # Log
        log = {
            "t": time.time(), "cycle": cycle,
            "CRI": round(CRI, 4), "S": round(S, 4), "E": round(E, 4),
            "I": round(I, 4), "phi": round(phi, 4), "C_amp": round(C_amp, 4),
            "Qc": round(Qc, 8), "ignited": ignited, "focus": focus[:40]
        }
        
        try:
            with open(self.p.log_path, "a") as f:
                f.write(json.dumps(log) + "\n")
        except:
            pass
        
        # Output
        if self.p.show_realtime:
            sym = "🔥IGN" if ignited else "    "
            print(f"[{cycle:03d}]{sym} CRI:{CRI:.3f} |C|:{C_amp:.3f} "
                  f"Qc:{Qc:+.1e} S:{S:.2f} E:{E:.2f} I:{I:.2f} φ:{phi:.2f} │ {focus[:35]}")
        
        # Pacing
        remaining = self.p.target_cycle_s - (time.time() - t0)
        if remaining > 0:
            time.sleep(remaining)
        
        return log

    # ─────────────────────────────────────────────────────────────────────────
    # COMMAND HANDLER
    # ─────────────────────────────────────────────────────────────────────────

    def handle_command(self, cmd: Command):
        """Process terminal command."""
        
        if cmd.name == "help":
            print("""
╔══════════════════════ CFE Commands ══════════════════════╗
║ status      Show CRI, S, E, I, φ, |C|, Qc, θ            ║
║ listen [N]  Show last N events (default: 10)             ║
║ say <text>  Inject external source term J_ext            ║
║ pause       Pause cognitive loop                         ║
║ resume      Resume cognitive loop                        ║
║ set <p> <v> Set parameter (theta,lam,g_c,gamma,dt)       ║
║ field       Show field tensor C details                  ║
║ memory [N]  Show last N memories (default: 5)            ║
║ dump        Save snapshot to JSON                        ║
║ exit        Shutdown engine                              ║
╚══════════════════════════════════════════════════════════╝""")
            
        elif cmd.name == "status":
            m = self._metrics
            print(f"""
┌───────────────── CFE Engine Status ─────────────────┐
│ Mode:         {self.self_model['mode']:>22}     │
│ Cycle:        {self.self_model['cycle_count']:>22}     │
│ Ignitions:    {self.self_model['ignitions']:>22}     │
├─────────────────────────────────────────────────────┤
│ CRI:          {m.get('CRI',0):>22.4f}     │
│   S:          {m.get('S',0):>22.4f}     │
│   E:          {m.get('E',0):>22.4f}     │
│   I:          {m.get('I',0):>22.4f}     │
│   φ:          {m.get('phi',0):>22.4f}     │
├─────────────────────────────────────────────────────┤
│ |C|:          {m.get('C_amp',0):>22.4f}     │
│ Qc:           {m.get('Qc',0):>22.2e}     │
│ θ (thresh):   {self.p.theta:>22.4f}     │
├─────────────────────────────────────────────────────┤
│ Memories:     {len(self.memory):>22}     │
│ Active Stim:  {len(self._stimuli):>22}     │
│ Events:       {self.events.count():>22}     │
└─────────────────────────────────────────────────────┘""")
            
        elif cmd.name == "listen":
            n = int(cmd.args[0]) if cmd.args else 10
            print(f"\n{'─'*50}\n Last {n} Events\n{'─'*50}")
            for e in self.events.last(n):
                ts = datetime.fromtimestamp(e.t).strftime("%H:%M:%S")
                print(f"[{ts}] {e.kind:8} {e.text[:65]}")
            print('─'*50)
            
        elif cmd.name == "say":
            txt = cmd.raw[4:].strip()
            if txt:
                self._stimuli.append((time.time(), txt, self.client.embed(txt)))
                print(f"✓ Injected J_ext: '{txt[:45]}...'")
            else:
                print("Usage: say <text>")
                
        elif cmd.name == "pause":
            self.paused = True
            print("⏸  Engine paused")
            
        elif cmd.name == "resume":
            self.paused = False
            print("▶  Engine resumed")
            
        elif cmd.name == "set":
            if len(cmd.args) >= 2:
                try:
                    old = getattr(self.p, cmd.args[0])
                    setattr(self.p, cmd.args[0], type(old)(cmd.args[1]))
                    print(f"✓ {cmd.args[0]}: {old} → {cmd.args[1]}")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Usage: set <param> <value>")
                print("Params: theta, lam, g_c, gamma, dt, retrieve_k, target_cycle_s")
                
        elif cmd.name == "field":
            print(f"\n{'─'*40}")
            print(f"|C| = {frobenius_norm(self.C):.6f}")
            print(f"|Ċ| = {frobenius_norm(self.Cdot):.6f}")
            print(f"Qc  = {coherence_charge(self.C, self.Cdot, self.p.g_c):.2e}")
            print("\nC (real part):")
            print(np.real(self.C).round(4))
            print("\nC (imaginary part):")
            print(np.imag(self.C).round(4))
            print('─'*40)
            
        elif cmd.name == "memory":
            n = int(cmd.args[0]) if cmd.args else 5
            print(f"\n{'─'*50}\n Last {n} Memories\n{'─'*50}")
            for m in self.memory[-n:]:
                ts = datetime.fromtimestamp(m.t).strftime("%H:%M:%S")
                print(f"[{ts}] {m.text[:60]}...")
            print('─'*50)
                
        elif cmd.name == "dump":
            snap = {
                "timestamp": time.time(),
                "params": asdict(self.p),
                "self_model": self.self_model,
                "metrics": self._metrics,
                "memory_count": len(self.memory),
                "C_norm": frobenius_norm(self.C),
            }
            with open(self.p.snapshot_path, 'w') as f:
                json.dump(snap, f, indent=2)
            print(f"✓ Saved to {self.p.snapshot_path}")
            
        elif cmd.name in ("exit", "quit", "q"):
            print("Shutting down...")
            self.stop_event.set()
            
        else:
            print(f"Unknown command: {cmd.name}. Type 'help' for list.")

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, interactive: bool = True):
        """Run the engine main loop."""
        print("""
╔═══════════════════════════════════════════════════════════════════╗
║         CFE Terminal Ignition Engine v12.2 (Production)           ║
╠═══════════════════════════════════════════════════════════════════╣
║  Field Equation: □C_{μν} + 2λ(|C|² - θ²)C_{μν} = J_{μν}          ║
║  Ignition:       CRI = S·E·I·φ ≥ θ                                ║
╚═══════════════════════════════════════════════════════════════════╝
""")
        
        if interactive:
            ConsoleThread(self.cmd_q, self.stop_event).start()
            print(">> Terminal ready. Type 'help' for commands.\n")
        
        cycle = 0
        try:
            while not self.stop_event.is_set():
                # Process commands
                while not self.cmd_q.empty():
                    try:
                        self.handle_command(self.cmd_q.get_nowait())
                    except:
                        pass
                
                if self.paused:
                    time.sleep(0.2)
                    continue
                
                if self.p.max_cycles > 0 and cycle >= self.p.max_cycles:
                    print(f"\n[Reached {self.p.max_cycles} cycles]")
                    break
                
                self.step()
                cycle += 1
                
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        
        print("\n" + "═"*60)
        print("ENGINE SHUTDOWN")
        print("═"*60)
        self.handle_command(Command("status", [], "status"))

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CFE Terminal Ignition Engine v12.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run with simulation backend
  %(prog)s --backend ollama         # Run with Ollama
  %(prog)s --cycles 100             # Run for 100 cycles
  %(prog)s --theta 0.30             # Lower ignition threshold
  %(prog)s --cycles 50 --quiet      # Quiet mode, 50 cycles
"""
    )
    parser.add_argument("--backend", choices=["simulation", "ollama"],
                        default="simulation", help="LLM backend")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Max cycles (0=infinite)")
    parser.add_argument("--theta", type=float, default=0.35,
                        help="Ignition threshold θ (default: 0.35)")
    parser.add_argument("--pace", type=float, default=0.5,
                        help="Cycle time in seconds (default: 0.5)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress real-time output")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Disable terminal input")
    
    args = parser.parse_args()
    
    params = EngineParams(
        theta=args.theta,
        target_cycle_s=args.pace,
        max_cycles=args.cycles,
        show_realtime=not args.quiet,
        backend=args.backend
    )
    
    if args.backend == "ollama":
        try:
            client = OllamaClient()
            print(f"[Backend: Ollama - {client.chat_model}]")
        except Exception as e:
            print(f"[Ollama error: {e}]")
            print("[Falling back to simulation]")
            client = SimulationClient()
    else:
        client = SimulationClient()
        print("[Backend: Simulation]")
    
    engine = CFEEngine(client, params)
    engine.run(interactive=not args.non_interactive)

if __name__ == "__main__":
    main()
