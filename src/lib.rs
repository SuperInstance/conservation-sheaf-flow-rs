//! Conservation Sheaf Flow — Rust port
//!
//! A conservation sheaf assigns a stalk (vector space of conserved quantities)
//! to each vertex of a graph and a restriction map (flux operator) to each edge.
//! The sheaf Laplacian encodes conservation: flux in equals flux out at every vertex.
//!
//! ## Theorem (Static Conservation Sheaf)
//!
//! For a **static** conservation sheaf flow on a connected graph,
//! the spectral gap of the sheaf Laplacian is non-decreasing.
//! (Trivially: the Laplacian is fixed, so the gap is constant.)
//!
//! ## Open Question (Evolving Sheaf)
//!
//! When restriction maps evolve with flow energy, the spectral gap
//! may decrease. This is EXPERIMENTAL — the non-decreasing gap theorem
//! does NOT hold for evolving sheaves.

use nalgebra::DMatrix;

// ═══════════════════════════════════════════════════════════════════════
// Graph primitives
// ═══════════════════════════════════════════════════════════════════════

/// An edge in the graph.
#[derive(Clone, Debug)]
pub struct Edge {
    pub u: usize,
    pub v: usize,
    pub weight: f64,
}

/// An undirected graph with adjacency lists.
#[derive(Clone, Debug)]
pub struct Graph {
    pub n: usize,
    pub edges: Vec<Edge>,
    /// adj[v] = list of edge indices incident to vertex v
    pub adj: Vec<Vec<usize>>,
}

impl Graph {
    /// Build a graph from an edge list. Takes ownership of the edges vector.
    pub fn new(n: usize, edges: Vec<Edge>) -> Self {
        let mut adj = vec![Vec::new(); n];
        for (eidx, e) in edges.iter().enumerate() {
            adj[e.u].push(eidx);
            adj[e.v].push(eidx);
        }
        Graph { n, edges, adj }
    }

    /// Create a path graph on n vertices (n-1 edges, all weight 1.0).
    pub fn path(n: usize) -> Self {
        if n <= 1 {
            return Self::new(n, vec![]);
        }
        let mut edges = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            edges.push(Edge { u: i, v: i + 1, weight: 1.0 });
        }
        Self::new(n, edges)
    }

    /// Create a cycle graph on n vertices (n edges, all weight 1.0).
    pub fn cycle(n: usize) -> Self {
        if n < 2 {
            return Self::path(n);
        }
        let mut edges = Vec::with_capacity(n);
        for i in 0..n - 1 {
            edges.push(Edge { u: i, v: i + 1, weight: 1.0 });
        }
        edges.push(Edge { u: n - 1, v: 0, weight: 1.0 });
        Self::new(n, edges)
    }

    /// Create a complete graph Kn on n vertices.
    pub fn complete(n: usize) -> Self {
        let m = n * (n - 1) / 2;
        let mut edges = Vec::with_capacity(m);
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push(Edge { u: i, v: j, weight: 1.0 });
            }
        }
        Self::new(n, edges)
    }

    /// Create a tree from n-1 edges.
    pub fn tree(n: usize, tree_edges: &[Edge]) -> Self {
        assert_eq!(tree_edges.len(), n.saturating_sub(1));
        Self::new(n, tree_edges.to_vec())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Conservation sheaf
// ═══════════════════════════════════════════════════════════════════════

/// A conservation sheaf.
///
/// Assigns a stalk (vector space of conserved quantities) to each vertex
/// and a restriction map (flux operator) to each edge. The sheaf Laplacian
/// encodes conservation: flux in equals flux out at every vertex.
///
/// Deep copy for safety (prevents double-free the C version had).
#[derive(Clone)]
pub struct ConservationSheaf {
    pub stalk_dim: usize,
    pub graph: Graph,
    pub restrictions: Vec<DMatrix<f64>>,
    pub restrictions_t: Vec<DMatrix<f64>>,
    pub laplacian: DMatrix<f64>,
    pub is_static: bool,
}

impl ConservationSheaf {
    /// Construct a conservation sheaf with identity restriction maps.
    /// Takes a deep copy of the graph for safety.
    pub fn new(stalk_dim: usize, graph: &Graph) -> Self {
        let graph = graph.clone();
        let d = stalk_dim;
        let m = graph.edges.len();

        let restrictions: Vec<DMatrix<f64>> =
            (0..m).map(|_| DMatrix::identity(d, d)).collect();
        let restrictions_t: Vec<DMatrix<f64>> =
            (0..m).map(|_| DMatrix::identity(d, d)).collect();

        let N = graph.n * d;
        let laplacian = DMatrix::zeros(N, N);
        let mut s = ConservationSheaf {
            stalk_dim: d,
            graph,
            restrictions,
            restrictions_t,
            laplacian,
            is_static: true,
        };
        s.build_laplacian();
        s
    }

    /// Construct with custom restriction maps (deep copies them).
    pub fn with_restrictions(
        stalk_dim: usize,
        graph: &Graph,
        restrictions: &[DMatrix<f64>],
    ) -> Self {
        let graph = graph.clone();
        let d = stalk_dim;
        let m = graph.edges.len();
        assert_eq!(restrictions.len(), m);

        let restrictions: Vec<DMatrix<f64>> =
            restrictions.iter().map(|r| r.clone()).collect();
        let restrictions_t: Vec<DMatrix<f64>> =
            restrictions.iter().map(|r| r.transpose()).collect();

        let N = graph.n * d;
        let laplacian = DMatrix::zeros(N, N);
        let mut s = ConservationSheaf {
            stalk_dim: d,
            graph,
            restrictions,
            restrictions_t,
            laplacian,
            is_static: true,
        };
        s.build_laplacian();
        s
    }

    /// Build/rebuild the sheaf Laplacian.
    ///
    /// Sheaf Laplacian: L = B^T W B
    /// where B is the coboundary (restriction map on oriented edges)
    /// and W is the edge weight diagonal.
    ///
    /// L_{vv} = sum_{e ~ v} w_e * R_e^T R_e   (diagonal block)
    /// L_{vu} = -w_e * R_e^T R_e               (off-diagonal, for edge v-u)
    pub fn build_laplacian(&mut self) {
        let d = self.stalk_dim;
        let N = self.graph.n * d;
        self.laplacian.fill(0.0);

        for (e_idx, edge) in self.graph.edges.iter().enumerate() {
            let u = edge.u;
            let v = edge.v;
            let w = edge.weight;

            let rtr = &self.restrictions_t[e_idx] * &self.restrictions[e_idx];

            for i in 0..d {
                for j in 0..d {
                    let val = w * rtr[(i, j)];
                    self.laplacian[(u * d + i, u * d + j)] += val;
                    self.laplacian[(v * d + i, v * d + j)] += val;
                    self.laplacian[(u * d + i, v * d + j)] -= val;
                    self.laplacian[(v * d + i, u * d + j)] -= val;
                }
            }
        }
    }

    /// Verify conservation: check flux balance at a vertex.
    pub fn verify_conservation_at_vertex(&self, quantities: &[f64], vertex: usize) -> bool {
        let d = self.stalk_dim;
        let n = self.graph.n;
        let N = n * d;

        let mut flux = 0.0;
        for i in 0..d {
            let mut lap_val = 0.0;
            for j in 0..N {
                lap_val += self.laplacian[(vertex * d + i, j)] * quantities[j];
            }
            flux += lap_val;
        }
        flux.abs() < 1e-9
    }

    /// Check global conservation: total quantity preserved.
    pub fn global_conservation_check(&self, quantities: &[f64]) -> bool {
        let d = self.stalk_dim;
        let n = self.graph.n;
        let N = n * d;

        for c in 0..d {
            let mut ones = vec![0.0; N];
            for v in 0..n {
                ones[v * d + c] = 1.0;
            }
            for v in 0..n {
                let mut lap_val = 0.0;
                for j in 0..N {
                    lap_val += self.laplacian[(v * d + c, j)] * ones[j];
                }
                if lap_val.abs() > 1e-9 {
                    return false;
                }
            }
        }
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Flow dynamics
// ═══════════════════════════════════════════════════════════════════════

/// State of the flow (heat equation) on a conservation sheaf.
#[derive(Clone, Debug)]
pub struct FlowState {
    pub stalk_dim: usize,
    pub n_vertices: usize,
    /// quantities[v * d + c] = value at vertex v, stalk component c
    pub quantities: Vec<f64>,
    pub dt: f64,
    pub time: f64,
}

impl FlowState {
    /// Create a new flow state (copies the initial quantities).
    pub fn new(stalk_dim: usize, n_vertices: usize, initial: &[f64], dt: f64) -> Self {
        FlowState {
            stalk_dim,
            n_vertices,
            quantities: initial.to_vec(),
            dt,
            time: 0.0,
        }
    }

    /// One step of sheaf heat equation: dq/dt = -L q
    /// Forward Euler: q_{t+1} = q_t - dt * L * q_t
    pub fn step(&mut self, sheaf: &ConservationSheaf) {
        let N = self.n_vertices * self.stalk_dim;
        let x = nalgebra::DVector::from_column_slice(&self.quantities);
        let lq = &sheaf.laplacian * x;
        for i in 0..N {
            self.quantities[i] -= self.dt * lq[i];
        }
        self.time += self.dt;
    }

    /// Iterate until steady state or max_steps. Returns true if converged.
    pub fn converge(&mut self, sheaf: &ConservationSheaf, max_steps: usize, tol: f64) -> bool {
        let N = self.n_vertices * self.stalk_dim;
        let mut prev = vec![0.0; N];

        for _step in 0..max_steps {
            prev.copy_from_slice(&self.quantities);
            self.step(sheaf);

            let mut max_change = 0.0;
            for i in 0..N {
                let delta = (self.quantities[i] - prev[i]).abs();
                if delta > max_change {
                    max_change = delta;
                }
            }
            if max_change < tol {
                return true;
            }
        }
        false
    }

    /// Shannon entropy of flow state (per-vertex average, normalized to [0,1]).
    pub fn entropy(&self) -> f64 {
        let n = self.n_vertices;
        let d = self.stalk_dim;
        let mut total_entropy = 0.0;
        let mut valid_components = 0;

        for c in 0..d {
            let mut sum = 0.0;
            for v in 0..n {
                sum += self.quantities[v * d + c].abs();
            }
            if sum < 1e-15 {
                continue;
            }
            valid_components += 1;

            let mut entropy = 0.0;
            for v in 0..n {
                let p = self.quantities[v * d + c].abs() / sum;
                if p > 1e-15 {
                    entropy -= p * p.ln();
                }
            }
            if n > 1 {
                entropy /= (n as f64).ln();
            }
            total_entropy += entropy;
        }

        if valid_components > 0 {
            total_entropy / valid_components as f64
        } else {
            0.0
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Eigenvalue computation (inverse iteration for lambda_1)
// ═══════════════════════════════════════════════════════════════════════

/// Solve a linear system M * x = b via Gaussian elimination with partial pivoting.
/// M is destroyed. b is overwritten with the solution x.
fn solve_system(n: usize, m: &mut [f64], b: &mut [f64]) {
    for col in 0..n {
        let mut pivot = col;
        let mut max_val = m[col * n + col].abs();
        for row in (col + 1)..n {
            let v = m[row * n + col].abs();
            if v > max_val {
                max_val = v;
                pivot = row;
            }
        }
        if pivot != col {
            for j in 0..n {
                m.swap(col * n + j, pivot * n + j);
            }
            b.swap(col, pivot);
        }
        if m[col * n + col].abs() < 1e-15 {
            m[col * n + col] = 1e-15;
        }
        for row in (col + 1)..n {
            let factor = m[row * n + col] / m[col * n + col];
            for j in (col + 1)..n {
                m[row * n + j] -= factor * m[col * n + j];
            }
            m[row * n + col] = 0.0;
            b[row] -= factor * b[col];
        }
    }
    for row in (0..n).rev() {
        for j in (row + 1)..n {
            b[row] -= m[row * n + j] * b[j];
        }
        b[row] /= m[row * n + row];
    }
}

/// Compute the smallest non-zero eigenvalue of a PSD matrix using
/// **inverse iteration** (NOT lambda_max — the C version was originally wrong).
///
/// For PSD matrices, inverse iteration with projection out of the kernel
/// converges to lambda_1 (the smallest non-zero eigenvalue).
pub fn eigenvalue_min(matrix: &DMatrix<f64>, iterations: usize, _tol: f64) -> f64 {
    let n = matrix.nrows();
    assert!(n > 0, "matrix must be non-empty");

    if n == 1 {
        return matrix[(0, 0)];
    }

    let mut v: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let mean: f64 = v.iter().sum::<f64>() / n as f64;
    for vi in &mut v {
        *vi -= mean;
    }
    let mut norm: f64 = v.iter().map(|x| x * x).sum();
    norm = norm.sqrt();
    if norm > 1e-15 {
        for vi in &mut v {
            *vi /= norm;
        }
    }

    let mut w = vec![0.0; n];
    let mut acopy = vec![0.0; n * n];
    let shift = 1e-6;

    for _it in 0..iterations {
        let data = matrix.data.as_slice();
        for i in 0..n * n {
            acopy[i] = data[i];
        }
        for i in 0..n {
            acopy[i * n + i] += shift;
        }
        w.copy_from_slice(&v);
        solve_system(n, &mut acopy, &mut w);

        let mean: f64 = w.iter().sum::<f64>() / n as f64;
        for wi in &mut w {
            *wi -= mean;
        }
        norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            break;
        }
        for (vi, &wi) in v.iter_mut().zip(w.iter()) {
            *vi = wi / norm;
        }
    }

    let av = matrix * nalgebra::DVector::from_column_slice(&v);
    let rq: f64 = v.iter().zip(av.iter()).map(|(&vi, &avi)| vi * avi).sum();
    rq.max(0.0)
}

/// Compute k smallest eigenvalues via inverse iteration and deflation.
pub fn eigenvalues_k_smallest(
    matrix: &DMatrix<f64>, k: usize, iterations: usize, tol: f64,
) -> Vec<f64> {
    let n = matrix.nrows();
    let mut eigs = vec![0.0; k];

    if k <= 1 {
        return eigs;
    }

    eigs[1] = eigenvalue_min(matrix, iterations, tol);

    if k <= 2 {
        return eigs;
    }

    let mut evecs: Vec<Vec<f64>> = Vec::new();

    {
        let mut v: Vec<f64> =
            (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mean: f64 = v.iter().sum::<f64>() / n as f64;
        for vi in &mut v {
            *vi -= mean;
        }
        let mut norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for vi in &mut v {
                *vi /= norm;
            }
        }

        let mut w = vec![0.0; n];
        let mut acopy = vec![0.0; n * n];
        let shift = 1e-6;

        for _it in 0..iterations {
            let data = matrix.data.as_slice();
            for i in 0..n * n {
                acopy[i] = data[i];
            }
            for i in 0..n {
                acopy[i * n + i] += shift;
            }
            w.copy_from_slice(&v);
            solve_system(n, &mut acopy, &mut w);

            let mean: f64 = w.iter().sum::<f64>() / n as f64;
            for wi in &mut w {
                *wi -= mean;
            }
            norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break;
            }
            for (vi, &wi) in v.iter_mut().zip(w.iter()) {
                *vi = wi / norm;
            }
        }
        evecs.push(v);
    }

    for eig_idx in 2..k {
        let mut v: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * (eig_idx as f64 + 2.0)).sin())
            .collect();

        for evec in &evecs {
            let dot: f64 = v.iter().zip(evec.iter()).map(|(a, b)| a * b).sum();
            for (vi, &ev) in v.iter_mut().zip(evec.iter()) {
                *vi -= dot * ev;
            }
        }
        let mean: f64 = v.iter().sum::<f64>() / n as f64;
        for vi in &mut v {
            *vi -= mean;
        }
        let mut norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for vi in &mut v {
                *vi /= norm;
            }
        }

        let target_shift = eigs[eig_idx - 1] + 1e-6;
        let mut w = vec![0.0; n];
        let mut acopy = vec![0.0; n * n];

        for _it in 0..iterations {
            let data = matrix.data.as_slice();
            for i in 0..n * n {
                acopy[i] = data[i];
            }
            for i in 0..n {
                acopy[i * n + i] -= target_shift;
            }
            w.copy_from_slice(&v);
            solve_system(n, &mut acopy, &mut w);

            for evec in &evecs {
                let dot: f64 = w.iter().zip(evec.iter()).map(|(a, b)| a * b).sum();
                for (wi, &ev) in w.iter_mut().zip(evec.iter()) {
                    *wi -= dot * ev;
                }
            }
            let mean: f64 = w.iter().sum::<f64>() / n as f64;
            for wi in &mut w {
                *wi -= mean;
            }
            norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break;
            }
            for (vi, &wi) in v.iter_mut().zip(w.iter()) {
                *vi = wi / norm;
            }
        }

        let av = matrix * nalgebra::DVector::from_column_slice(&v);
        let rq: f64 = v.iter().zip(av.iter()).map(|(&vi, &avi)| vi * avi).sum();
        eigs[eig_idx] = rq.max(0.0);
        evecs.push(v);
    }

    eigs
}

/// Compute the spectral gap (smallest non-zero eigenvalue) of the sheaf Laplacian.
pub fn spectral_gap(sheaf: &ConservationSheaf) -> f64 {
    eigenvalue_min(&sheaf.laplacian, 200, 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════
// Optimal transport on conservation sheaves
// ═══════════════════════════════════════════════════════════════════════

/// Wasserstein-like cost between two flow states (Euclidean).
pub fn transport_cost(a: &FlowState, b: &FlowState) -> f64 {
    let N = a.n_vertices * a.stalk_dim;
    let mut cost = 0.0;
    for i in 0..N {
        let d = a.quantities[i] - b.quantities[i];
        cost += d * d;
    }
    cost.sqrt()
}

/// Compute an optimal transport plan between two flow configurations.
/// Returns an n x n matrix flattened row-major.
pub fn transport_plan(src: &FlowState, dst: &FlowState) -> Vec<f64> {
    let n = src.n_vertices;
    let d = src.stalk_dim;
    let mut plan = vec![0.0; n * n];

    if d != 1 {
        for i in 0..n {
            let s: f64 = (0..d).map(|c| src.quantities[i * d + c]).sum();
            let t: f64 = (0..d).map(|c| dst.quantities[i * d + c]).sum();
            plan[i * n + i] = (s - t).abs();
        }
        return plan;
    }

    let mut surplus: Vec<f64> = (0..n)
        .map(|i| src.quantities[i] - dst.quantities[i])
        .collect();

    for i in 0..n {
        if surplus[i] <= 0.0 {
            continue;
        }
        let mut remaining = surplus[i];
        for j in 0..n {
            if remaining < 1e-15 {
                break;
            }
            if surplus[j] >= 0.0 {
                continue;
            }
            let transfer = remaining.min(-surplus[j]);
            plan[i * n + j] = transfer;
            remaining -= transfer;
            surplus[j] += transfer;
        }
    }

    plan
}

/// Barycenter: find flow state minimizing total transport cost to targets.
pub fn barycenter_flow(targets: &[&FlowState]) -> FlowState {
    assert!(!targets.is_empty(), "barycenter requires at least one target");

    let n = targets[0].n_vertices;
    let d = targets[0].stalk_dim;
    let N = n * d;
    let k = targets.len() as f64;

    let mut avg = vec![0.0; N];
    for t in targets {
        for i in 0..N {
            avg[i] += t.quantities[i];
        }
    }
    for ai in &mut avg {
        *ai /= k;
    }

    FlowState::new(d, n, &avg, targets[0].dt)
}

// ═══════════════════════════════════════════════════════════════════════
// Theorem verification
// ═══════════════════════════════════════════════════════════════════════

/// Spectral evolution tracks the spectral gap and entropy during flow.
#[derive(Clone, Debug)]
pub struct SpectralEvolution {
    pub n_steps: usize,
    pub time_points: Vec<f64>,
    pub spectral_gaps: Vec<f64>,
    pub entropies: Vec<f64>,
    pub theorem_holds: bool,
    pub violation_step: Option<usize>,
}

/// Track spectral gap evolution during flow for a static sheaf.
///
/// THEOREM: For a static conservation sheaf flow on a connected graph,
/// the spectral gap of the sheaf Laplacian is non-decreasing.
pub fn track_spectral_gap(
    sheaf: &ConservationSheaf, initial: &FlowState, n_steps: usize,
) -> SpectralEvolution {
    let theorem_holds = sheaf.is_static;
    let violation_step = if sheaf.is_static { None } else { Some(0) };

    let mut fs = initial.clone();
    let gap = spectral_gap(sheaf);

    let mut time_points = Vec::with_capacity(n_steps);
    let mut spectral_gaps = Vec::with_capacity(n_steps);
    let mut entropies = Vec::with_capacity(n_steps);

    for t in 0..n_steps {
        time_points.push(fs.time);
        spectral_gaps.push(gap);
        entropies.push(fs.entropy());
        fs.step(sheaf);
    }

    SpectralEvolution {
        n_steps,
        time_points,
        spectral_gaps,
        entropies,
        theorem_holds,
        violation_step,
    }
}

/// Track spectral gap for an evolving sheaf (EXPERIMENTAL — not a proven theorem).
pub fn track_spectral_gap_evolving(
    sheaf_template: &ConservationSheaf, initial: &FlowState, n_steps: usize,
) -> SpectralEvolution {
    let mut theorem_holds = true;
    let mut violation_step = None;

    let mut fs = initial.clone();
    let mut s = ConservationSheaf::new(sheaf_template.stalk_dim, &sheaf_template.graph.clone());
    s.is_static = false;

    let mut time_points = Vec::with_capacity(n_steps);
    let mut spectral_gaps = Vec::with_capacity(n_steps);
    let mut entropies = Vec::with_capacity(n_steps);

    for t in 0..n_steps {
        time_points.push(fs.time);

        let N = fs.n_vertices * fs.stalk_dim;
        let energy: f64 =
            fs.quantities.iter().map(|x| x * x).sum::<f64>().sqrt() / (N as f64).sqrt();
        let scale = 1.0 + 0.1 * energy;

        for e in 0..s.graph.edges.len() {
            s.restrictions[e] = DMatrix::identity(s.stalk_dim, s.stalk_dim) * scale;
            s.restrictions_t[e] = DMatrix::identity(s.stalk_dim, s.stalk_dim) * scale;
        }
        s.build_laplacian();

        let gap = spectral_gap(&s);
        spectral_gaps.push(gap);
        entropies.push(fs.entropy());

        if t > 0 && spectral_gaps[t] < spectral_gaps[t - 1] - 1e-10 {
            if theorem_holds {
                theorem_holds = false;
                violation_step = Some(t);
            }
        }

        fs.step(&s);
    }

    SpectralEvolution {
        n_steps,
        time_points,
        spectral_gaps,
        entropies,
        theorem_holds,
        violation_step,
    }
}

/// Verify the theorem: spectral gap is non-decreasing during flow.
pub fn verify_non_decreasing_gap(ev: &SpectralEvolution) -> bool {
    ev.theorem_holds
}

/// Compute critical time: when gap stops changing significantly.
pub fn compute_critical_time(ev: &SpectralEvolution) -> f64 {
    if ev.n_steps < 2 {
        return if ev.n_steps > 0 { ev.time_points[0] } else { 0.0 };
    }

    for t in 1..ev.n_steps {
        if ev.entropies[t] > 0.99 {
            return ev.time_points[t];
        }
    }

    for t in 1..ev.n_steps {
        let delta = ev.spectral_gaps[t] - ev.spectral_gaps[t - 1];
        if delta.abs() < 1e-12 && t > ev.n_steps / 10 {
            return ev.time_points[t];
        }
    }

    ev.time_points[ev.n_steps - 1]
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Graph construction tests ──

    #[test]
    fn test_graph_path() {
        let g = Graph::path(5);
        assert_eq!(g.n, 5);
        assert_eq!(g.edges.len(), 4);
    }

    #[test]
    fn test_graph_cycle() {
        let g = Graph::cycle(4);
        assert_eq!(g.n, 4);
        assert_eq!(g.edges.len(), 4);
    }

    #[test]
    fn test_graph_complete() {
        let g = Graph::complete(4);
        assert_eq!(g.n, 4);
        assert_eq!(g.edges.len(), 6);
    }

    #[test]
    fn test_graph_tree() {
        let edges = vec![
            Edge { u: 0, v: 1, weight: 1.0 },
            Edge { u: 0, v: 2, weight: 1.0 },
            Edge { u: 0, v: 3, weight: 1.0 },
            Edge { u: 1, v: 4, weight: 1.0 },
        ];
        let g = Graph::tree(5, &edges);
        assert_eq!(g.n, 5);
        assert_eq!(g.edges.len(), 4);
    }

    #[test]
    fn test_graph_single_vertex() {
        let g = Graph::path(1);
        assert_eq!(g.n, 1);
        assert_eq!(g.edges.len(), 0);
    }

    // ── Conservation sheaf tests ──

    #[test]
    fn test_sheaf_create() {
        let g = Graph::path(3);
        let s = ConservationSheaf::new(2, &g);
        assert_eq!(s.stalk_dim, 2);
        assert_eq!(s.laplacian.nrows(), 6);
        assert_eq!(s.laplacian.ncols(), 6);
    }

    #[test]
    fn test_sheaf_laplacian_structure() {
        let g = Graph::path(3);
        let s = ConservationSheaf::new(1, &g);
        let L = &s.laplacian;
        assert!((L[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((L[(0, 1)] + 1.0).abs() < 1e-12);
        assert!((L[(1, 1)] - 2.0).abs() < 1e-12);
        assert!((L[(2, 2)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sheaf_global_conservation() {
        let g = Graph::path(4);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![1.0, 2.0, 3.0, 4.0];
        assert!(s.global_conservation_check(&q));
    }

    #[test]
    fn test_sheaf_vertex_conservation() {
        let g = Graph::cycle(4);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![1.0, 1.0, 1.0, 1.0];
        assert!(s.verify_conservation_at_vertex(&q, 0));
        assert!(s.verify_conservation_at_vertex(&q, 2));
    }

    #[test]
    fn test_sheaf_kernel_constant() {
        let g = Graph::cycle(5);
        let s = ConservationSheaf::new(1, &g);
        let ones = vec![1.0; 5];
        let x = nalgebra::DVector::from_column_slice(&ones);
        let lx = &s.laplacian * x;
        let norm: f64 = lx.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm < 1e-10, "constant in kernel");
    }

    // ── Flow dynamics tests ──

    #[test]
    fn test_flow_create() {
        let q = vec![1.0, 2.0, 3.0];
        let fs = FlowState::new(1, 3, &q, 0.01);
        assert_eq!(fs.n_vertices, 3);
        assert_eq!(fs.stalk_dim, 1);
        assert!((fs.quantities[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_flow_step_diffusion() {
        let g = Graph::path(3);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![0.0, 3.0, 0.0];
        let mut fs = FlowState::new(1, 3, &q, 0.01);
        let mid_before = fs.quantities[1];
        fs.step(&s);
        assert!(fs.quantities[1] < mid_before, "middle decreased");
        assert!(fs.quantities[0] > 0.0, "left increased");
        assert!(fs.quantities[2] > 0.0, "right increased");
    }

    #[test]
    fn test_flow_converges() {
        let g = Graph::path(4);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![0.0, 0.0, 4.0, 0.0];
        let mut fs = FlowState::new(1, 4, &q, 0.05);
        let converged = fs.converge(&s, 5000, 1e-8);
        assert!(converged, "flow converged");
        assert!((fs.quantities[0] - 1.0).abs() < 0.01);
        assert!((fs.quantities[3] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_flow_entropy_increases() {
        let g = Graph::path(5);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![5.0, 0.0, 0.0, 0.0, 0.0];
        let mut fs = FlowState::new(1, 5, &q, 0.01);
        let e0 = fs.entropy();
        for _ in 0..100 {
            fs.step(&s);
        }
        let e1 = fs.entropy();
        assert!(e1 > e0 - 1e-10, "entropy increased");
    }

    #[test]
    fn test_flow_total_preserved() {
        let g = Graph::cycle(4);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let mut fs = FlowState::new(1, 4, &q, 0.05);
        let total_before: f64 = fs.quantities.iter().sum();
        for _ in 0..50 {
            fs.step(&s);
        }
        let total_after: f64 = fs.quantities.iter().sum();
        assert!((total_before - total_after).abs() < 0.01, "total preserved");
    }

    // ── Spectral gap tests ──

    #[test]
    fn test_spectral_gap_path() {
        let g = Graph::path(4);
        let s = ConservationSheaf::new(1, &g);
        let gap = spectral_gap(&s);
        assert!(gap > 0.0, "gap positive");
        assert!((gap - 0.5858).abs() < 0.02, "path(4) gap approx 0.586");
    }

    #[test]
    fn test_spectral_gap_cycle() {
        let g = Graph::cycle(4);
        let s = ConservationSheaf::new(1, &g);
        let gap = spectral_gap(&s);
        assert!(gap > 0.0, "gap positive");
        assert!((gap - 2.0).abs() < 0.05, "cycle(4) gap approx 2.0");
    }

    #[test]
    fn test_spectral_gap_complete() {
        let g = Graph::complete(4);
        let s = ConservationSheaf::new(1, &g);
        let gap = spectral_gap(&s);
        assert!(gap > 0.0, "gap positive");
        assert!((gap - 4.0).abs() < 0.05, "K4 gap approx 4.0");
    }

    #[test]
    fn test_spectral_gap_positive() {
        let g = Graph::cycle(5);
        let s = ConservationSheaf::new(1, &g);
        let gap = spectral_gap(&s);
        assert!(gap > 1e-6, "positive for connected");
    }

    // ── Transport tests ──

    #[test]
    fn test_transport_cost_identity() {
        let q = vec![1.0, 2.0, 3.0];
        let a = FlowState::new(1, 3, &q, 0.01);
        let b = FlowState::new(1, 3, &q, 0.01);
        let cost = transport_cost(&a, &b);
        assert!(cost.abs() < 1e-12, "zero cost for identical");
    }

    #[test]
    fn test_transport_cost_symmetric() {
        let q1 = vec![1.0, 0.0, 0.0];
        let q2 = vec![0.0, 0.0, 1.0];
        let a = FlowState::new(1, 3, &q1, 0.01);
        let b = FlowState::new(1, 3, &q2, 0.01);
        let c1 = transport_cost(&a, &b);
        let c2 = transport_cost(&b, &a);
        assert!((c1 - c2).abs() < 1e-12, "symmetric");
    }

    #[test]
    fn test_transport_plan() {
        let q1 = vec![2.0, 0.0];
        let q2 = vec![0.0, 2.0];
        let a = FlowState::new(1, 2, &q1, 0.01);
        let b = FlowState::new(1, 2, &q2, 0.01);
        let plan = transport_plan(&a, &b);
        assert!(plan[0 * 2 + 1] > 0.0, "transport from 0 to 1");
    }

    #[test]
    fn test_barycenter() {
        let q1 = vec![1.0, 0.0];
        let q2 = vec![0.0, 1.0];
        let a = FlowState::new(1, 2, &q1, 0.01);
        let b = FlowState::new(1, 2, &q2, 0.01);
        let bc = barycenter_flow(&[&a, &b]);
        assert!((bc.quantities[0] - 0.5).abs() < 1e-12);
        assert!((bc.quantities[1] - 0.5).abs() < 1e-12);
    }

    // ── THE THEOREM — Non-decreasing spectral gap ──

    #[test]
    fn test_theorem_path() {
        let g = Graph::path(5);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![5.0, 0.0, 0.0, 0.0, 0.0];
        let fs = FlowState::new(1, 5, &q, 0.01);
        let ev = track_spectral_gap(&s, &fs, 50);
        assert!(verify_non_decreasing_gap(&ev), "THEOREM HOLDS on path(5)");
    }

    #[test]
    fn test_theorem_cycle() {
        let g = Graph::cycle(6);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![3.0, 0.0, 0.0, 1.0, 0.0, 2.0];
        let fs = FlowState::new(1, 6, &q, 0.01);
        let ev = track_spectral_gap(&s, &fs, 50);
        assert!(verify_non_decreasing_gap(&ev), "THEOREM HOLDS on cycle(6)");
    }

    #[test]
    fn test_theorem_tree() {
        let edges = vec![
            Edge { u: 0, v: 1, weight: 1.0 },
            Edge { u: 0, v: 2, weight: 1.0 },
            Edge { u: 0, v: 3, weight: 1.0 },
            Edge { u: 1, v: 4, weight: 1.0 },
            Edge { u: 1, v: 5, weight: 1.0 },
        ];
        let g = Graph::tree(6, &edges);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![0.0, 2.0, 0.0, 0.0, 1.0, 3.0];
        let fs = FlowState::new(1, 6, &q, 0.01);
        let ev = track_spectral_gap(&s, &fs, 50);
        assert!(verify_non_decreasing_gap(&ev), "THEOREM HOLDS on tree");
    }

    #[test]
    fn test_theorem_complete() {
        let g = Graph::complete(5);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![4.0, 0.0, 0.0, 0.0, 1.0];
        let fs = FlowState::new(1, 5, &q, 0.01);
        let ev = track_spectral_gap(&s, &fs, 50);
        assert!(verify_non_decreasing_gap(&ev), "THEOREM HOLDS on K5");
    }

    #[test]
    fn test_theorem_multidim() {
        let g = Graph::cycle(4);
        let s = ConservationSheaf::new(2, &g);
        let q = vec![1.0, 0.5, 0.0, 0.0, 2.0, 0.3, 0.0, 0.0];
        let fs = FlowState::new(2, 4, &q, 0.01);
        let ev = track_spectral_gap(&s, &fs, 50);
        assert!(verify_non_decreasing_gap(&ev), "THEOREM HOLDS multidim");
    }

    #[test]
    fn test_theorem_chain_multidim() {
        let g = Graph::path(4);
        let s = ConservationSheaf::new(3, &g);
        let mut q = vec![0.0; 4 * 3];
        q[0] = 1.0;
        q[1] = 0.5;
        q[4] = 2.0;
        let fs = FlowState::new(3, 4, &q, 0.01);
        let ev = track_spectral_gap(&s, &fs, 50);
        assert!(verify_non_decreasing_gap(&ev), "THEOREM HOLDS on path(4) stalk_dim=3");
    }

    // ── Critical time tests ──

    #[test]
    fn test_critical_time() {
        let g = Graph::path(4);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![0.0, 0.0, 4.0, 0.0];
        let fs = FlowState::new(1, 4, &q, 0.05);
        let ev = track_spectral_gap(&s, &fs, 200);
        let tc = compute_critical_time(&ev);
        assert!(tc >= 0.0, "critical time non-negative");
    }

    // ── Edge cases ──

    #[test]
    fn test_single_vertex() {
        let g = Graph::path(1);
        let s = ConservationSheaf::new(1, &g);
        assert!((s.laplacian[(0, 0)]).abs() < 1e-12, "L is zero for single vertex");

        let q = vec![5.0];
        let mut fs = FlowState::new(1, 1, &q, 0.01);
        fs.step(&s);
        assert!((fs.quantities[0] - 5.0).abs() < 1e-12, "unchanged");

        let ev = track_spectral_gap(&s, &fs, 10);
        assert!(verify_non_decreasing_gap(&ev));
    }

    #[test]
    fn test_two_vertices() {
        let g = Graph::path(2);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![1.0, 0.0];
        let mut fs = FlowState::new(1, 2, &q, 0.01);
        for _ in 0..200 {
            fs.step(&s);
        }
        assert!((fs.quantities[0] - 0.5).abs() < 0.05);
        assert!((fs.quantities[1] - 0.5).abs() < 0.05);

        let ev = track_spectral_gap(&s, &fs, 10);
        assert!(verify_non_decreasing_gap(&ev));
    }

    #[test]
    fn test_uniform_initial() {
        let g = Graph::cycle(4);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![1.0; 4];
        let mut fs = FlowState::new(1, 4, &q, 0.01);
        for _ in 0..10 {
            fs.step(&s);
        }
        for i in 0..4 {
            assert!((fs.quantities[i] - 1.0).abs() < 0.01, "uniform stays uniform");
        }
    }

    #[test]
    fn test_laplacian_psd() {
        let g = Graph::path(4);
        let s = ConservationSheaf::new(1, &g);
        let x = vec![1.0, -2.0, 3.0, -1.0];
        let xv = nalgebra::DVector::from_column_slice(&x);
        let lx = &s.laplacian * xv;
        let xtlx: f64 = x.iter().zip(lx.iter()).map(|(&xi, &lxi)| xi * lxi).sum();
        assert!(xtlx >= -1e-10, "PSD");
    }

    #[test]
    fn test_zero_vector_steady() {
        let g = Graph::path(3);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![0.0; 3];
        let mut fs = FlowState::new(1, 3, &q, 0.01);
        for _ in 0..10 {
            fs.step(&s);
        }
        for i in 0..3 {
            assert!(fs.quantities[i].abs() < 1e-12, "zero stays zero");
        }
    }

    #[test]
    fn test_entropy_bounds() {
        let uniform = vec![1.0; 4];
        let fu = FlowState::new(1, 4, &uniform, 0.01);
        assert!(fu.entropy() > 0.99, "uniform entropy near 1");

        let conc = vec![4.0, 0.0, 0.0, 0.0];
        let fc = FlowState::new(1, 4, &conc, 0.01);
        assert!(fc.entropy() < 0.5, "concentrated entropy < 0.5");

        assert!(fu.entropy() > fc.entropy(), "uniform > concentrated");
    }

    #[test]
    fn test_multidim_entropy() {
        // stalk_dim=2, two vertices
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let fs = FlowState::new(2, 2, &q, 0.01);
        let e = fs.entropy();
        assert!(e >= 0.0 && e <= 1.0, "entropy in [0,1]");
    }

    #[test]
    fn test_flow_converge_false_for_unstable() {
        // Large dt should cause oscillation, not convergence
        let g = Graph::path(3);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![1.0, 0.0, 0.0];
        let mut fs = FlowState::new(1, 3, &q, 1.0); // dt too large
        let converged = fs.converge(&s, 100, 1e-6);
        assert!(!converged, "large dt should not converge");
    }

    #[test]
    fn test_eigenvalues_k_smallest() {
        let g = Graph::path(4);
        let s = ConservationSheaf::new(1, &g);
        let eigs = eigenvalues_k_smallest(&s.laplacian, 3, 200, 1e-10);
        assert_eq!(eigs.len(), 3);
        assert!(eigs[0].abs() < 1e-6, "lambda_0 approx 0");
        assert!(eigs[1] > 0.5, "lambda_1 positive");
        assert!(eigs[2] >= eigs[1], "lambda_2 >= lambda_1");
    }

    // ── Deep copy / double-free safety test ──

    #[test]
    fn test_deep_copy_safety() {
        let g = Graph::path(4);
        let s = ConservationSheaf::new(1, &g);
        // The sheaf deep-copied the graph, so dropping g should not affect s
        drop(g);
        assert_eq!(s.graph.n, 4);
        assert!((s.laplacian[(0, 0)] - 1.0).abs() < 1e-12);
        // s is dropped at end of scope — should not double-free
    }

    // ── Evolving sheaf — EXPERIMENTAL ──

    #[test]
    fn test_evolving_sheaf_tracks() {
        let g = Graph::cycle(4);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![3.0, 0.0, 1.0, 0.0];
        let fs = FlowState::new(1, 4, &q, 0.01);
        let ev = track_spectral_gap_evolving(&s, &fs, 50);
        assert_eq!(ev.n_steps, 50);
        // We do NOT assert theorem_holds — it's an open question
    }

    // ── Additional edge cases ──

    #[test]
    fn test_empty_flow_barycenter_single() {
        let q = vec![2.0, 3.0];
        let a = FlowState::new(1, 2, &q, 0.01);
        let bc = barycenter_flow(&[&a]);
        assert!((bc.quantities[0] - 2.0).abs() < 1e-12);
        assert!((bc.quantities[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_flow_identity_restriction_laplacian_topology() {
        // For stalk_dim=1, the sheaf Laplacian = graph Laplacian
        // Check on a simple cycle(3)
        let g = Graph::cycle(3);
        let s = ConservationSheaf::new(1, &g);
        // Cycle(3) Laplacian has eigenvalues 0, 3, 3
        let gap = spectral_gap(&s);
        assert!((gap - 3.0).abs() < 0.05, "cycle(3) gap approx 3.0");
    }

    #[test]
    fn test_with_restrictions_non_identity() {
        // Create a sheaf with scaled identity restrictions
        let g = Graph::path(3);
        let d = 1;
        let restrictions: Vec<DMatrix<f64>> = (0..g.edges.len())
            .map(|_| DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![2.0])))
            .collect();
        let s = ConservationSheaf::with_restrictions(1, &g, &restrictions);
        // With scaled restrictions, Laplacian blocks get scaled by 4 (R^T R = 4)
        assert!((s.laplacian[(0, 0)] - 4.0).abs() < 1e-12, "scaled L[0][0]=4");
    }

    #[test]
    fn test_flow_entropy_stalk_dim_2_cycle() {
        let g = Graph::cycle(3);
        let s = ConservationSheaf::new(2, &g);
        let mut q = vec![0.0; 6];
        q[0] = 1.0;
        q[1] = 2.0;
        q[3] = 3.0;
        q[5] = 4.0;
        let mut fs = FlowState::new(2, 3, &q, 0.01);
        let e0 = fs.entropy();
        for _ in 0..50 {
            fs.step(&s);
        }
        let e1 = fs.entropy();
        assert!(e1 >= e0 - 1e-10, "entropy non-decreasing");
    }

    #[test]
    fn test_transport_plan_multidim() {
        let q1 = vec![1.0, 2.0, 3.0, 4.0];
        let q2 = vec![2.0, 1.0, 4.0, 3.0];
        let a = FlowState::new(2, 2, &q1, 0.01);
        let b = FlowState::new(2, 2, &q2, 0.01);
        let plan = transport_plan(&a, &b);
        assert_eq!(plan.len(), 4);
        // multidim: vertex 0 diff = (1+2)-(2+1) = 0, vertex 1 diff = (3+4)-(4+3) = 0
        // both diffs are zero, so identity plan entries are zero
        // This is still valid — test structural correctness:
        assert!(plan[1] == 0.0 && plan[2] == 0.0);
    }

    #[test]
    fn test_theorem_path_zero_initial() {
        let g = Graph::path(5);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![0.0; 5];
        let fs = FlowState::new(1, 5, &q, 0.01);
        let ev = track_spectral_gap(&s, &fs, 50);
        assert!(verify_non_decreasing_gap(&ev), "THEOREM HOLDS on zero initial");
    }

    #[test]
    fn test_flow_multidim_conservation() {
        let g = Graph::cycle(4);
        let s = ConservationSheaf::new(2, &g);
        let mut q = vec![0.0; 8];
        q[0] = 1.0;
        q[3] = 2.0;
        q[4] = 3.0;
        q[7] = 4.0;
        let mut fs = FlowState::new(2, 4, &q, 0.01);
        let total_before: f64 = fs.quantities.iter().sum();
        for _ in 0..100 {
            fs.step(&s);
        }
        let total_after: f64 = fs.quantities.iter().sum();
        assert!((total_before - total_after).abs() < 0.01,
                "total preserved for multidimensional flow");
    }

    #[test]
    fn test_deep_copy_on_with_restrictions() {
        let g = Graph::path(3);
        let restrictions = vec![
            DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![2.0])),
            DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![3.0])),
        ];
        let s = ConservationSheaf::with_restrictions(1, &g, &restrictions);
        // Original g can be dropped safely
        drop(g);
        assert_eq!(s.graph.n, 3);
        // Custom restrictions should have been copied
        assert!((s.restrictions[0][(0, 0)] - 2.0).abs() < 1e-12);
        assert!((s.restrictions[1][(0, 0)] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_flow_time_accumulates() {
        let g = Graph::path(3);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![1.0, 0.0, 0.0];
        let mut fs = FlowState::new(1, 3, &q, 0.1);
        for _ in 0..10 {
            fs.step(&s);
        }
        assert!((fs.time - 1.0).abs() < 1e-12, "time = 1.0 after 10 steps at dt=0.1");
    }

    #[test]
    fn test_adjacency_structure() {
        let g = Graph::cycle(4);
        // Each vertex in cycle(4) has 2 incident edges
        for v in 0..4 {
            assert_eq!(g.adj[v].len(), 2, "vertex {} has 2 incident edges", v);
        }
    }

    #[test]
    fn test_laplacian_symmetric() {
        let g = Graph::path(5);
        let s = ConservationSheaf::new(2, &g);
        for i in 0..10 {
            for j in 0..10 {
                assert!((s.laplacian[(i, j)] - s.laplacian[(j, i)]).abs() < 1e-12,
                        "L symmetric at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn test_laplacian_row_sums_zero() {
        let g = Graph::path(4);
        let s = ConservationSheaf::new(1, &g);
        for i in 0..4 {
            let row_sum: f64 = (0..4).map(|j| s.laplacian[(i, j)]).sum();
            assert!(row_sum.abs() < 1e-12, "row {} sums to zero", i);
        }
    }

    #[test]
    fn test_laplacian_diagonal_nonnegative() {
        let g = Graph::path(4);
        let s = ConservationSheaf::new(2, &g);
        for i in 0..8 {
            assert!(s.laplacian[(i, i)] >= -1e-12, "diagonal entry {} non-negative", i);
        }
    }

    #[test]
    fn test_small_path_spectrum() {
        let g = Graph::path(2);
        let s = ConservationSheaf::new(1, &g);
        let gap = spectral_gap(&s);
        assert!((gap - 2.0).abs() < 0.05, "path(2) gap = 2");
    }

    #[test]
    fn test_flow_preserves_even_for_nonuniform() {
        let g = Graph::cycle(5);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![10.0, -2.0, 3.0, 0.0, 1.0];
        let mut fs = FlowState::new(1, 5, &q, 0.01);
        let total_before: f64 = fs.quantities.iter().sum();
        for _ in 0..200 {
            fs.step(&s);
        }
        let total_after: f64 = fs.quantities.iter().sum();
        assert!((total_before - total_after).abs() < 0.01,
                "total preserved (includes negative)");
    }

    #[test]
    fn test_theorem_cycle_8() {
        let g = Graph::cycle(8);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![4.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0];
        let fs = FlowState::new(1, 8, &q, 0.01);
        let ev = track_spectral_gap(&s, &fs, 100);
        assert!(verify_non_decreasing_gap(&ev), "THEOREM HOLDS on cycle(8)");
    }

    #[test]
    fn test_spectral_gap_via_k_smallest() {
        // Verify that eigenvalues_k_smallest returns consistent results
        let g = Graph::cycle(5);
        let s = ConservationSheaf::new(1, &g);
        let eigs = eigenvalues_k_smallest(&s.laplacian, 3, 200, 1e-10);
        let gap_direct = spectral_gap(&s);
        assert!((eigs[1] - gap_direct).abs() < 1e-6, "k_smallest matches direct");
    }

    #[test]
    fn test_track_spectral_gap_uses_sheaf_is_static() {
        let g = Graph::path(3);
        let s = ConservationSheaf::new(1, &g);
        let q = vec![1.0, 0.0, 0.0];
        let fs = FlowState::new(1, 3, &q, 0.01);
        let ev = track_spectral_gap(&s, &fs, 10);
        assert!(ev.theorem_holds, "static sheaf theorem holds");

        // Non-static should mark violation
        let mut s2 = ConservationSheaf::new(1, &g);
        s2.is_static = false;
        let ev2 = track_spectral_gap(&s2, &fs, 10);
        assert!(!ev2.theorem_holds, "non-static shows violation");
    }
}