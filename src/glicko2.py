from dataclasses import dataclass
import math
from typing import Optional, Tuple

# Constants
MU_DEFAULT = 1500.0
RD_DEFAULT = 350.0
SIGMA_DEFAULT = 0.06
TAU = 0.5  # System constant, constraints change of volatility over time
EPSILON = 0.000001 # Convergence tolerance

# Glicko-2 Scaling Factor
GLICKO2_SCALE = 173.7178

@dataclass
class Glicko2Rating:
    mu: float = MU_DEFAULT
    rd: float = RD_DEFAULT
    sigma: float = SIGMA_DEFAULT
    last_match_day: Optional[int] = None
    
class Glicko2Tracker:
    def __init__(self):
        # Key: (discipline, player_name) -> Glicko2Rating
        self.ratings: dict[Tuple[str, str], Glicko2Rating] = {}

    def get_rating(self, discipline: str, player: str) -> Glicko2Rating:
        key = (discipline, player)
        if key not in self.ratings:
            self.ratings[key] = Glicko2Rating()
        return self.ratings[key]
    
    def get_mu(self, discipline: str, player: str) -> float:
        return self.get_rating(discipline, player).mu
        
    def get_rd(self, discipline: str, player: str) -> float:
        return self.get_rating(discipline, player).rd
        
    def get_sigma(self, discipline: str, player: str) -> float:
        return self.get_rating(discipline, player).sigma

    def update(self, discipline: str, winner: str, loser: str, match_day: int) -> None:
        r_w = self.get_rating(discipline, winner)
        r_l = self.get_rating(discipline, loser)
        
        # Apply time-based RD inflation first
        _inflate_rd(r_w, match_day)
        _inflate_rd(r_l, match_day)
        
        # Calculate new ratings
        new_w = _calculate_new_rating(r_w, r_l, 1.0)
        new_l = _calculate_new_rating(r_l, r_w, 0.0)
        
        # Update state
        self.ratings[(discipline, winner)] = new_w
        self.ratings[(discipline, loser)] = new_l

def _inflate_rd(player: Glicko2Rating, current_day: int) -> None:
    if player.last_match_day is None:
        player.last_match_day = current_day
        return

    days_inactive = current_day - player.last_match_day
    if days_inactive > 0:
        # Convert to Glicko-2 scale
        phi = player.rd / GLICKO2_SCALE
        # New Phi (c = 1 assumed for daily scale, simplified here)
        phi_new = math.sqrt(phi**2 + (player.sigma**2 * days_inactive))
        # Convert back and cap
        player.rd = min(phi_new * GLICKO2_SCALE, 350.0)
    
    player.last_match_day = current_day

def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / (math.pi**2))

def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))

def _calculate_new_rating(player: Glicko2Rating, opponent: Glicko2Rating, score: float) -> Glicko2Rating:
    # 1. Convert to Glicko-2 scale
    mu = (player.mu - MU_DEFAULT) / GLICKO2_SCALE
    phi = player.rd / GLICKO2_SCALE
    sigma = player.sigma
    
    mu_j = (opponent.mu - MU_DEFAULT) / GLICKO2_SCALE
    phi_j = opponent.rd / GLICKO2_SCALE
    
    # 2. Compute v (estimated variance) based on game outcome
    g_phi_j = _g(phi_j)
    E_val = _E(mu, mu_j, phi_j)
    v = 1.0 / (g_phi_j**2 * E_val * (1.0 - E_val))
    
    # 3. Compute Delta (estimated improvement)
    delta = v * g_phi_j * (score - E_val)
    
    # 4. Update volatility (sigma) - Iterative algorithm
    a = math.log(sigma**2)
    def f(x):
        ex = math.exp(x)
        num1 = ex * (delta**2 - phi**2 - v - ex)
        den1 = 2 * ((phi**2 + v + ex)**2)
        term2 = (x - a) / (TAU**2)
        return (num1 / den1) - term2
        
    # Illinois algorithm for root finding not strictly needed if we just do standard Newton-Raphson or simple bisection
    # Let's use a simplified approach for stability as per Glicko paper
    A = a
    if delta**2 > phi**2 + v:
        B = math.log(delta**2 - phi**2 - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        B = a - k * TAU
        
    fA = f(A)
    fB = f(B)
    
    while abs(B - A) > EPSILON:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB < 0:
            A = B
            fA = fB
        else:
            fA = fA / 2.0
        B = C
        fB = fC
        
    sigma_prime = math.exp(A / 2.0)
    
    # 5. Update Rating and RD
    phi_star = math.sqrt(phi**2 + sigma_prime**2)
    phi_prime = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
    mu_prime = mu + phi_prime**2 * g_phi_j * (score - E_val)
    
    # 6. Convert back to Glicko-1 scale
    return Glicko2Rating(
        mu = mu_prime * GLICKO2_SCALE + MU_DEFAULT,
        rd = phi_prime * GLICKO2_SCALE,
        sigma = sigma_prime,
        last_match_day = player.last_match_day
    )
