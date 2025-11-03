# IN-RIL Integration: Gradient Projection for Conflict Resolution

* **IN-RIL gradient projection** integrated into SAC actor with **minimal code changes**,
* **conflict-aware projection** for resolving gradient interference between RL and BC objectives,
* **surrogate loss construction** for standard optimizer compatibility,
* **configuration knobs** with safe defaults,
* **mathematical foundation** aligned with gradient surgery principles.

---

# IN-RIL: Gradient Projection for Conflict Resolution

This integration adds **IN-RIL-style gradient projection** to your SAC implementation, enabling the policy to resolve conflicts between reinforcement learning and behavior cloning objectives through intelligent gradient surgery. The implementation uses **conflict detection** via cosine similarity and **orthogonal projection** to remove interfering gradient components.

## What IN-RIL Adds (One-Screen Summary)

- **Conflict detection**: Cosine similarity between RL and BC gradients
- **Gradient projection**: Removes conflicting components using orthogonal projection
- **Two projection modes**:
  - `"bc_on_rl_orth"`: Projects BC gradients onto RL orthogonal complement
  - `"mutual"`: PCGrad-style mutual projection for both objectives
- **Surrogate loss**: Constructs scalar loss whose gradient equals projected direction
- **Zero changes** to your training loop - all integration is internal to the policy

---

## Mathematical Foundation

### 1.1 SAC Actor Objective (Baseline)

Your standard SAC actor minimizes:
$$\mathcal{L}_{\text{SAC}}(\theta) = \mathbb{E}[\alpha \log \pi_\theta(a'|s) - \min_i Q_i(s,a')]$$

### 1.2 FlowRL Weighted BC Term

FlowRL adds a **weighted behavior cloning** term:
$$\mathcal{L}_{\text{FlowBC}}(\theta) = \mathbb{E}[f(Q^{\pi_{\beta^*}}(s,a) - Q^{\pi_\theta}(s,a')) \cdot \|a' - a\|^2]$$

where $f(x) = \max(x, 0)$ is the ReLU weight function.

### 1.3 Gradient Conflict Detection

Define the per-step gradients w.r.t. **actor** parameters:
$$g_{\text{RL}} = \nabla_\theta \mathcal{L}_{\text{SAC}}, \quad g_{\text{BC}} = \nabla_\theta (\lambda_{\text{BC}} \mathcal{L}_{\text{FlowBC}})$$

**Conflict** is detected via cosine similarity:
$$\cos(g_{\text{RL}}, g_{\text{BC}}) = \frac{\langle g_{\text{RL}}, g_{\text{BC}} \rangle}{|g_{\text{RL}}| \cdot |g_{\text{BC}}|}$$

When $\cos(g_{\text{RL}}, g_{\text{BC}}) < \tau_{\text{conf}}$ (default $\tau_{\text{conf}} = 0$), the gradients conflict.

### 1.4 Gradient Projection

**Mode 1: BC onto RL orthogonal complement (default)**
$$\tilde{g}_{\text{BC}} = \begin{cases}
g_{\text{BC}} - \frac{\langle g_{\text{BC}}, g_{\text{RL}} \rangle}{|g_{\text{RL}}|^2 + \epsilon} g_{\text{RL}}, & \text{if } \cos(g_{\text{RL}}, g_{\text{BC}}) < \tau_{\text{conf}} \\
g_{\text{BC}}, & \text{otherwise}
\end{cases}$$

**Mode 2: Mutual projection (PCGrad-style)**
When in conflict, project both gradients:
$$\tilde{g}_{\text{RL}} = g_{\text{RL}} - \frac{\langle g_{\text{RL}}, g_{\text{BC}} \rangle}{|g_{\text{BC}}|^2 + \epsilon} g_{\text{BC}}$$
$$\tilde{g}_{\text{BC}} = g_{\text{BC}} - \frac{\langle g_{\text{BC}}, \tilde{g}_{\text{RL}} \rangle}{|\tilde{g}_{\text{RL}}|^2 + \epsilon} \tilde{g}_{\text{RL}}$$

### 1.5 Final Gradient Direction

$$g_{\text{final}} = \tilde{g}_{\text{RL}} + \tilde{g}_{\text{BC}}$$

### 1.6 Surrogate Loss Construction

To use with standard optimizers, construct a **surrogate scalar loss** whose gradient equals $g_{\text{final}}$:

$$\mathcal{L}_{\text{surrogate}}(\theta) = \sum_j \langle \theta_j, (g_{\text{final}})_j^{\perp} \rangle$$

where $(g_{\text{final}})_j^{\perp}$ is **detached** from the computation graph.

**Return value for logging:**
$$\mathcal{L}_{\text{return}} = \mathcal{L}_{\text{surrogate}} + (\mathcal{L}_{\text{sum}} - \mathcal{L}_{\text{surrogate}})^{\text{detach}}$$

This ensures `.item()` shows the unprojected scalar while `.backward()` uses the projected gradient.

---

## Implementation Details

### 2.1 Architecture Changes

**New components** (only created if `flowrl_gradproj_enabled=True`):
- `_actor_trainable_params()` → Get actor parameters matching optimizer selection
- `_dot()`, `_norm2()`, `_scale()`, `_sub()`, `_add()` → Gradient vector operations
- `_surrogate_loss_from_grads()` → Construct surrogate loss from projected gradients

**No optimizer changes needed:**
- All gradient projection happens inside `compute_loss_actor()`
- Standard `loss.backward()` receives projected gradients via surrogate loss

### 2.2 Actor Training (`compute_loss_actor`)

**Standard SAC + FlowRL:** unchanged loss computation

**Added gradient projection:**
```python
# Compute gradients for each objective
g_rl = torch.autograd.grad(L_rl, params, retain_graph=True, create_graph=False, allow_unused=True)
g_bc = torch.autograd.grad(lambda_bc * L_bc, params, retain_graph=True, create_graph=False, allow_unused=True)

# Conflict detection
cos = dot(g_rl, g_bc) / (norm(g_rl) * norm(g_bc) + eps)
conflict = (cos < threshold)

# Projection based on mode
if mode == "mutual":
    # PCGrad-style mutual projection
    if conflict:
        g_rl = g_rl - proj_coeff * g_bc
        g_bc = g_bc - proj_coeff * g_rl
else:
    # Default: project BC onto RL orthogonal complement
    if conflict:
        g_bc = g_bc - proj_coeff * g_rl

# Construct surrogate loss
g_final = g_rl + g_bc
L_surrogate = surrogate_loss_from_grads(params, g_final)
```

### 2.3 Gradient Vector Operations

**Robust handling of None gradients:**
- `_dot()`: Computes dot product, handles None gradients gracefully
- `_norm2()`: Computes squared norm, handles None gradients
- `_scale()`, `_sub()`, `_add()`: Element-wise operations with None handling

**Numerical stability:**
- Epsilon regularization: `eps = 1e-12` for norm computations
- Cosine clamping: `cos.clamp(-1.0, 1.0)`
- NaN detection: Raises error if surrogate loss is NaN

### 2.4 Logging Integration

**Optional diagnostics** (enabled automatically):
```python
self._actor_proj_stats = {
    "actor_proj/cos": cos.detach().mean().item(),
    "actor_proj/conflict_frac": conflict.float().detach().mean().item(),
    "actor_proj/mode": 0 if mode=="bc_on_rl_orth" else 1,
}
```

**Learner integration:**
```python
# In learner.py - automatically logs projection stats
if hasattr(policy, "_actor_proj_stats"):
    training_infos.update(policy._actor_proj_stats)
```

---

## Configuration & Usage

### 3.1 Configuration Parameters

Add to your `SACConfig`:

```python
# IN-RIL gradient projection parameters
flowrl_gradproj_enabled: bool = False                   # Enable gradient projection
flowrl_gradproj_mode: str = "bc_on_rl_orth"             # "bc_on_rl_orth" or "mutual"
flowrl_gradproj_conflict_cos_thresh: float = 0.0        # Conflict detection threshold
flowrl_gradproj_eps: float = 1e-8                       # Numerical stability epsilon
```

**Safe defaults:** All parameters have safe defaults; missing configs won't crash.

### 3.2 Example Configuration

```yaml
# Enable FlowRL + IN-RIL projection
flow_rl_enabled: true
flow_rl_bc_weight: 0.1

# IN-RIL gradient projection
flowrl_gradproj_enabled: true
flowrl_gradproj_mode: "bc_on_rl_orth"
flowrl_gradproj_conflict_cos_thresh: 0.0
flowrl_gradproj_eps: 1e-12

# Your existing SAC config...
policy:
  type: "sac_flowrl"  # Use FlowRL policy with projection
  # ... other SAC parameters
```

### 3.3 Training Usage

**No changes to your training script!** Once enabled:

- **Actor updates:** minimize SAC loss + weighted BC with gradient projection
- **Conflict resolution:** automatic when gradients interfere
- **Inference:** `select_action()` works unchanged
- **Evaluation:** standard policy evaluation

---

## Practical Recommendations

### 4.1 Hyperparameter Tuning

**Conflict threshold $\tau_{\text{conf}}$:**
- $\tau_{\text{conf}} = 0.0$: Project when gradients are orthogonal or opposing
- $\tau_{\text{conf}} < 0$: More aggressive projection (earlier intervention)
- $\tau_{\text{conf}} > 0$: Only project when gradients are strongly opposing

**Projection mode:**
- `"bc_on_rl_orth"` (default): Preserves RL signal, removes conflicting BC components
- `"mutual"`: PCGrad-style, reduces interference in both directions

**Numerical stability:**
- `eps = 1e-12` is usually sufficient
- Increase if you see numerical issues in gradient norms

### 4.2 Training Tips

**Monitoring projection activity:**
- Track `actor_proj/cos`: Should be negative when conflicts occur
- Track `actor_proj/conflict_frac`: Fraction of batches with conflicts
- Monitor gradient norms: Should remain stable

**When to use which mode:**
- **`bc_on_rl_orth`**: When RL is primary objective, BC is regularization
- **`mutual`**: When both objectives are equally important

**Troubleshooting:**
- **"Surrogate loss is NaN"**: Check for numerical issues, increase `eps`
- **"No conflicts detected"**: Check if BC weight is too small
- **"Projection too aggressive"**: Increase conflict threshold

### 4.3 Performance Considerations

**Computational cost:**
- **2× gradient computation**: Two `autograd.grad()` calls per actor update
- **Memory**: `retain_graph=True` keeps computation graph temporarily
- **No optimizer changes**: Standard backward pass works unchanged

**When projection helps:**
- **Early training**: When RL and BC objectives conflict
- **High BC weights**: When behavior cloning dominates
- **Complex tasks**: When objectives have different optimal directions

---

## API Reference

### 4.1 Policy Class

```python
class SACFlowRLPolicy(SACPolicy):
    name = "sac_flowrl"
    
    # IN-RIL gradient projection (enabled if flowrl_gradproj_enabled=True)
    def _actor_trainable_params(self) -> List[Parameter]
    def _dot(self, gs1: List[Tensor], gs2: List[Tensor]) -> Tensor
    def _norm2(self, gs: List[Tensor]) -> Tensor
    def _scale(self, gs: List[Tensor], c: float) -> List[Tensor]
    def _sub(self, gs_a: List[Tensor], gs_b: List[Tensor]) -> List[Tensor]
    def _add(self, gs_a: List[Tensor], gs_b: List[Tensor]) -> List[Tensor]
    def _surrogate_loss_from_grads(self, params: List[Parameter], grads: List[Tensor]) -> Tensor
```

### 4.2 Key Methods

```python
# Gradient projection computation
g_rl = torch.autograd.grad(L_rl, params, retain_graph=True, create_graph=False, allow_unused=True)
g_bc = torch.autograd.grad(lambda_bc * L_bc, params, retain_graph=True, create_graph=False, allow_unused=True)

# Conflict detection
cos = policy._dot(g_rl, g_bc) / (policy._norm2(g_rl) * policy._norm2(g_bc) + eps)
conflict = (cos < threshold)

# Projection
if conflict:
    g_bc = policy._sub(g_bc, policy._scale(g_rl, proj_coeff))

# Surrogate loss
L_surrogate = policy._surrogate_loss_from_grads(params, g_final)
```

### 4.3 Configuration Registration

The policy is registered as:
```python
@PreTrainedConfig.register_subclass("sac")
@PreTrainedConfig.register_subclass("sac_flowrl")  # Same config, different name
class SACConfig(PreTrainedConfig):
    # ... IN-RIL parameters
    flowrl_gradproj_enabled: bool = False
    flowrl_gradproj_mode: str = "bc_on_rl_orth"
    flowrl_gradproj_conflict_cos_thresh: float = 0.0
    flowrl_gradproj_eps: float = 1e-12
```

---

## Mathematical References

### 5.1 Gradient Surgery Principles

- **PCGrad**: *Gradient Surgery for Multi-Task Learning*, Yu et al., 2020. ([arXiv](https://arxiv.org/abs/2001.06782))
- **GradNorm**: *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*, Chen et al., 2018. ([arXiv](https://arxiv.org/abs/1711.02257))
- **Conflict detection**: Cosine similarity between gradient vectors

### 5.2 Implementation Notes

**Stop-gradient conventions:**
- Projected gradients are **detached** in surrogate loss construction
- Original loss values are **detached** for logging consistency
- No gradient flow through projection operations to prevent circular dependencies

**Shape consistency:**
- All gradient operations handle **None gradients** gracefully
- Surrogate loss is **scalar** (sum over all parameters)
- Conflict detection is **element-wise** then **mean-reduced**

**Numerical stability:**
- **Epsilon regularization** on all norm computations
- **Cosine clamping** to prevent numerical overflow
- **NaN detection** with clear error messages

---

## What We Didn't Implement (By Design)

- **No multi-task projection**: Focuses on RL vs BC conflict only
- **No adaptive thresholds**: Fixed conflict detection threshold
- **No gradient clipping**: Relies on projection for stability
- **No momentum integration**: Standard SGD/Adam optimizers work unchanged

The gradient projection is designed to be **minimal and focused** - resolving the specific conflict between RL exploration and BC exploitation objectives.

---

## References

- **PCGrad paper**: *Gradient Surgery for Multi-Task Learning*, Yu et al., 2020. ([arXiv](https://arxiv.org/abs/2001.06782))
- **GradNorm paper**: *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*, Chen et al., 2018. ([arXiv](https://arxiv.org/abs/1711.02257))
- **FlowRL paper**: *Flow-Based Policy for Online Reinforcement Learning*, Lv et al., 2025. ([ar5iv](https://ar5iv.org/pdf/2506.12811))
- **Gradient surgery**: General principles for resolving gradient conflicts in multi-objective optimization
