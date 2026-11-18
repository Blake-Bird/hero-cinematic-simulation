/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief GPU offloading for spectral CR diffusion (OpenACC / OpenMP target / CPU-OMP fallback).
 *
 * Pattern follows OpenACC/conduction.hpp. Key difference from conduction: the matrix-vector
 * product cr_diffusion_matrix_multiply is called up to ~300 times per energy bin, so the
 * compute_diag flag is threaded through to avoid recomputing the row-sum diagonal on every call.
 *
 * @author OpenGadget3-development team, in particular Ludwig Böss.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifdef ACC_OPENACC
#include <openacc.h>
#elif defined(ACC_OPENMP)
#include <omp.h>
#endif // ACC_OPENACC

ACC_DECLARE_ROUTINE_BEFORE
int ngb_treefind_pairs_threads_local(MyLongDouble *searchcenter, MyFloat hsml, int mode, int *startnode,
                                     struct gravity_walkdata *aP, int *ngblist, int maxnumngb, int i_node);
ACC_DECLARE_ROUTINE_AFTER

ACC_DECLARE_ROUTINE_BEFORE
int cr_diffusion_evaluate_local(int target, int p_target, int mode, double *in, MyFloat *Kappa, MyFloat *KappaB,
                                int *ngblist, int extended_numngb_inbox, struct cr_diffusiondata_in *CRDiffusionDataGet,
                                struct cr_diffusiondata_out *CRDiffusionDataResult, struct gravity_walkdata *P,
                                struct cr_diffusiondata_in *SphP, int node, int i_node, int compute_diag);
ACC_DECLARE_ROUTINE_AFTER

// Bin-level GPU setup. Called once per bin after the Kappa loop closes; allocates
// and uploads the per-bin invariants (LiteCIn, CRDiffusionActiveParticleList, Kappa, KappaB).
// LiteCOut and `in[0:N_gas]` change every cr_diffusion_matrix_multiply call and are managed
// per-call inside cr_diffusion_matrix_multiply itself.
inline void openacc_cr_diffusion_bin_setup()
{
    ENDRUN_IF((acc_all.go_gpu != 0 && acc_all.go_gpu != 1), "acc_all.go_gpu should be 0 or 1 ", acc_all.go_gpu);

    CRDiffusionDoPrimaryOnGPU = acc_all.go_gpu;
    CRDiffusionShouldDrift = !acc_all.go_gpu;

    // Allocate NumPart entries so neighbor indices j < N_gas are always valid
    LiteCIn = (struct cr_diffusiondata_in *)mymalloc("LiteCIn", NumPart * sizeof(struct cr_diffusiondata_in));
    CRDiffusionActiveParticleList = (int *)mymalloc("DAPL", N_gas * sizeof(int));

    NCRDiffusionActivePart = 0;
    if (CRDiffusionDoPrimaryOnGPU)
    {
#pragma omp parallel for
        for (int i = 0; i < N_gas; i++)
        {
            particle2in_cr_diffusion(&LiteCIn[i], i);
        }

        // All gas particles participate in CR diffusion. Build the active list as the
        // identity mapping CRDiffusionActiveParticleList[i] = i. cr_diffusion_evaluate_local
        // does p_target = target and the post-merge does out2particle_cr_diffusion(...,i,1),
        // both of which require this identity. An atomic-capture build would scramble order.
#pragma omp parallel for
        for (int i = 0; i < N_gas; i++)
        {
            CRDiffusionActiveParticleList[i] = i;
        }
        NCRDiffusionActivePart = N_gas;
        CRDiffusionDoPrimaryOnGPU = NCRDiffusionActivePart > ACC_ACTIVEPART_CPU;
    }

    if (CRDiffusionDoPrimaryOnGPU)
    {
#if defined(ACC_OPENACC)
#pragma acc enter data create(CRDiffusionActiveParticleList [0:NCRDiffusionActivePart], LiteCIn [0:NumPart],           \
                              KAPPAB_ZERO_NGAS Kappa [0:N_gas])
#pragma acc update device(CRDiffusionActiveParticleList [0:NCRDiffusionActivePart], LiteCIn [0:NumPart],               \
                          KAPPAB_ZERO_NGAS Kappa [0:N_gas])
#elif defined(ACC_OPENMP)
#pragma omp target enter data map(alloc                                                                                \
                                  : CRDiffusionActiveParticleList [0:NCRDiffusionActivePart], LiteCIn [0:NumPart],     \
                                    KAPPAB_ZERO_NGAS Kappa [0:N_gas])
#pragma omp target update to(CRDiffusionActiveParticleList [0:NCRDiffusionActivePart], LiteCIn [0:NumPart],            \
                             KAPPAB_ZERO_NGAS Kappa [0:N_gas])
#endif // defined(ACC_OPENACC) || defined(ACC_OPENMP)
    }
}

// Bin-level GPU teardown. Called once per bin before its myfree; mirrors
// openacc_cr_diffusion_bin_setup. The free order respects mymalloc's LIFO discipline
// (LiteCIn and CRDiffusionActiveParticleList are the most recently allocated buffers).
inline void openacc_cr_diffusion_bin_teardown()
{
    if (CRDiffusionDoPrimaryOnGPU)
    {
#if defined(ACC_OPENACC)
#pragma acc exit data delete (CRDiffusionActiveParticleList [0:NCRDiffusionActivePart], LiteCIn [0:NumPart],           \
                              KAPPAB_ZERO_NGAS Kappa [0:N_gas])
#elif defined(ACC_OPENMP)
#pragma omp target exit data map(delete                                                                                \
                                 : CRDiffusionActiveParticleList [0:NCRDiffusionActivePart], LiteCIn [0:NumPart],      \
                                   KAPPAB_ZERO_NGAS Kappa [0:N_gas])
#endif // defined(ACC_OPENACC) || defined(ACC_OPENMP)
    }

    myfree(CRDiffusionActiveParticleList);
    myfree(LiteCIn);
    CRDiffusionActiveParticleList = NULL;
    LiteCIn = NULL;
    NCRDiffusionActivePart = 0;
    CRDiffusionDoPrimaryOnGPU = 0;
}

inline void openacc_cr_diffusion_primary_phase(int DoPrimaryOnGPU, int NCRDiffusionActivePart,
                                               int *CRDiffusionActiveParticleList,
                                               struct cr_diffusiondata_out *LiteCOut,
                                               struct cr_diffusiondata_in *LiteCIn, double *in, MyFloat *Kappa,
                                               MyFloat *KappaB, int compute_diag)
{
    if (DoPrimaryOnGPU)
    {
        int MiniNGBList[ACC_GPU_NGB_CR_DIFFUSION];
        for (int i = 0; i < ACC_GPU_NGB_CR_DIFFUSION; i++)
        {
            MiniNGBList[i] = 0;
        }
        VERBOSE(3, "CRDiffusion on GPU NumPart=%d N_gas=%d NActivePart=%d", NumPart, N_gas, NCRDiffusionActivePart);

#if defined(ACC_OPENACC)
#pragma acc parallel loop present(WalkLiteP [0:N_gas], COPY_KAPPAB Kappa [0:N_gas], in [0:N_gas]) private(MiniNGBList) \
    async(1)
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for private(MiniNGBList) nowait
#elif defined(ACC_NOACC)
#pragma omp parallel for private(MiniNGBList)
#endif // ACC_NOACC
        for (int il = 0; il < NCRDiffusionActivePart; il++)
        {
            int p_target = CRDiffusionActiveParticleList[il];
            int startnode = All.MaxPart;
            int endnode = startnode;
            int i_node = 0;

            while (endnode >= 0)
            {
                startnode = endnode;
                int numngb =
                    ngb_treefind_pairs_threads_local(WalkLiteP[p_target].Pos, WalkLiteP[p_target].Hsml, 2, &endnode,
                                                     WalkLiteP, MiniNGBList, ACC_GPU_NGB_CR_DIFFUSION, i_node);
                int ret = cr_diffusion_evaluate_local(il, p_target, 2, in, Kappa, KappaB, MiniNGBList, numngb, NULL,
                                                      LiteCOut, WalkLiteP, LiteCIn, i_node, startnode, compute_diag);
                i_node++;
            }
        } // end parallel for
    } // end DoPrimaryOnGPU

    VERBOSE(3, "CRDiffusion on GPU NumPart=%d N_gas=%d NActivePart=%d done", NumPart, N_gas, NCRDiffusionActivePart);
}

inline void openacc_cr_diffusion_secondary_phase(struct cr_diffusiondata_in *CRDiffusionDataGet,
                                                 struct cr_diffusiondata_out *CRDiffusionDataResult,
                                                 struct cr_diffusiondata_in *LiteCIn, int DoPrimaryOnGPU,
                                                 int should_drift, double *in, MyFloat *Kappa, MyFloat *KappaB,
                                                 double *out, double *sum, int compute_diag)
{
    int DoSecondaryOnGPU = DoPrimaryOnGPU && (Nimport > ACC_ACTIVEPART_II_CPU);
#ifdef ACC_II_TEST
    DoSecondaryOnGPU = DoSecondaryOnGPU && acc_async_test(1);
#endif
#if defined(ACC_OPENMP)
    DoSecondaryOnGPU = 0;
#endif // ACC_OPENMP

    if (!DoSecondaryOnGPU)
    {
        VERBOSE(1, " ar_cr_diffusion_evaluate_secondary(%d) Nimport=%d loop on CPU", !acc_all.go_gpu, Nimport);
        // CRDiffusionDataResult comes from mymalloc, which does not zero. The tree
        // walk writes every import row it visits, but a row whose node list is
        // exhausted before a single evaluate call is never written, and its
        // uninitialised Out/Sum would be sent back and merged into out[]/diag[].
        // The GPU branch below relies on its own per-row memset for the same reason.
        memset(CRDiffusionDataResult, 0, (size_t)Nimport * sizeof(struct cr_diffusiondata_out));
#pragma omp parallel
        {
            struct cr_thread_params cr_threadparams;
            cr_threadparams.tid = omp_get_thread_num();
            cr_threadparams.in = in;
            cr_threadparams.out = out;
            cr_threadparams.sum = sum;
            cr_threadparams.compute_diag = compute_diag;

            // Dispatch on the compute_diag template parameter.
            if (compute_diag)
                ar_green_evaluate_secondary(CRDiffusionDataGet, cr_diffusion_evaluate<true>, Module_CRDiffusion,
                                            TreeWalkType_GreenTreeWalk, NeighbourSearchCriteria_Hydro, &cr_threadparams,
                                            should_drift);
            else
                ar_green_evaluate_secondary(CRDiffusionDataGet, cr_diffusion_evaluate<false>, Module_CRDiffusion,
                                            TreeWalkType_GreenTreeWalk, NeighbourSearchCriteria_Hydro, &cr_threadparams,
                                            should_drift);
        }
    }
    else if (Nimport > 0)
    { // do secondary on GPU
        {
            int MiniNGBListII[ACC_GPU_NGB_CR_DIFFUSION];
            TIMER_PUSH(1, "ACC_II");
#if defined(ACC_OPENACC)
#pragma acc wait(1)
#pragma acc parallel loop copyin(CRDiffusionDataGet [0:Nimport], DataNodeListGet [0:Nimport])                          \
    present(LiteCIn [0:N_gas], in [0:N_gas], COPY_KAPPAB Kappa [0:N_gas])                                              \
        copyout(CRDiffusionDataResult [0:Nimport]) private(MiniNGBListII)
#elif defined(ACC_OPENMP) // ACC_OPENACC
#pragma omp taskwait
#pragma omp target teams distribute parallel for private(MiniNGBListII)                                                \
    map(to                                                                                                             \
        : CRDiffusionDataGet [0:Nimport], DataNodeListGet [0:Nimport]) map(from                                        \
                                                                           : CRDiffusionDataResult [0:Nimport])
#endif // ACC_OPENMP

            for (int i = 0; i < Nimport; i++)
            {
                memset_gpu((unsigned char *)&CRDiffusionDataResult[i], 0, sizeof(struct cr_diffusiondata_out));

                int i_node = 0;

                for (int node_list_i = 0; node_list_i < NODELISTLENGTH; node_list_i++)
                {
                    if (DataNodeListGet[i].NodeList[node_list_i] == -1)
                        break;
                    int startnode = DataNodeListGet[i].NodeList[node_list_i];
                    int endnode = startnode;
                    int first_iter = 1;

                    while (endnode >= 0)
                    {
                        startnode = endnode;

                        int numngb = ngb_treefind_pairs_threads_local(
                            CRDiffusionDataGet[i].Pos, CRDiffusionDataGet[i].Hsml, 1, &endnode, WalkLiteP,
                            MiniNGBListII, ACC_GPU_NGB_CR_DIFFUSION, first_iter);
                        int ret = cr_diffusion_evaluate_local(i, i, 1, in, Kappa, KappaB, MiniNGBListII, numngb,
                                                              CRDiffusionDataGet, CRDiffusionDataResult, WalkLiteP,
                                                              LiteCIn, i_node, startnode, compute_diag);
                        i_node++;
                        first_iter = 0;
                    } // end while
                } // end for nodes
            } // end parallel for
            TIMER_POP(1, "ACC_II");
        } // end secondary on GPU
    } // end DoSecondaryOnGPU
}

#ifdef CR_DIFF_DEVICE_SOLVE
// ============================================================================
// GPU SpMV solve path (CR_DIFF_DEVICE_SOLVE = ACC_SPECTRAL_CRs_DIFFUSION).
//
// The bin-independent geometry is built once per timestep on the CPU (the
// geometry tree walk records CR_CSR), made device-resident once, and every
// per-bin matrix is recombined ON THE DEVICE; the ENTIRE fused 2-RHS BiCGStab
// solve then runs there. Conventions follow the rest of this file: data is
// staged with `enter data`/`update`, and kernels reference the (file-static)
// arrays directly — no per-kernel map clause on the OpenMP path, explicit
// present(...) on the OpenACC path. ACC_NOACC runs everything on the host
// (plain OpenMP) so the algorithm can be validated without a GPU.
//
// The off-rank coupling matches the matrix-free path: each rank computes the
// contribution of its LOCAL in[j] to the REMOTE rows it was asked to serve
// (dcr_imp_*), exchanges those partial sums (per import/export slot), and
// scatter-adds the received values into its local rows (dcr_exp_place). Every
// halo is exchanged through host buffers — MPICH on this platform cannot take a
// device allocation of this size — so each one is bracketed by chunked staging.
// ============================================================================

// ---------------------------------------------------------------------------
// Chunked host<->device staging.
//
// One `target update` of a multi-GB array section is refused by the offload
// runtime ("Failure to copy data from device to host ... dataRetrieveImpl -1"):
// the whole transfer has to be backed by a single staging host block, and the
// largest request is the first to be denied once that pool is fragmented by the
// per-timestep map/unmap of this module's working set. The halo dimensions are
// sums over the comm-loop iterations, so the FCT buffers reach several GB per
// transfer on a full-active step.
//
// Splitting a section into CR_STAGE_CHUNK_BYTES pieces moves the same bytes in
// the same order and leaves the values bit-identical; only the size of each
// request to the runtime changes. The element counts are size_t so the byte
// offsets cannot wrap on a large halo.
#define CR_STAGE_CHUNK_BYTES ((size_t)256 * 1024 * 1024)

template <typename T> inline void cr_stage_from_device(T *p, size_t n)
{
#if defined(ACC_OPENACC) || defined(ACC_OPENMP)
    const size_t chunk = CR_STAGE_CHUNK_BYTES / sizeof(T);
    for (size_t off = 0; off < n; off += chunk)
    {
        const size_t len = (n - off) < chunk ? (n - off) : chunk;
#if defined(ACC_OPENACC)
#pragma acc update self(p [off:len])
#else
#pragma omp target update from(p [off:len])
#endif
    }
#else
    (void)p;
    (void)n; // ACC_NOACC: the array IS the host storage
#endif
}

template <typename T> inline void cr_stage_to_device(T *p, size_t n)
{
#if defined(ACC_OPENACC) || defined(ACC_OPENMP)
    const size_t chunk = CR_STAGE_CHUNK_BYTES / sizeof(T);
    for (size_t off = 0; off < n; off += chunk)
    {
        const size_t len = (n - off) < chunk ? (n - off) : chunk;
#if defined(ACC_OPENACC)
#pragma acc update device(p [off:len])
#else
#pragma omp target update to(p [off:len])
#endif
    }
#else
    (void)p;
    (void)n;
#endif
}

// ---------------------------------------------------------------------------
// Nonblocking halo machinery. The build's do-while comm structure is captured as
// a flat message list (one entry per (iter, rank) pair with traffic, offsets
// absolute into the flattened buffers). Every halo — the per-matvec result-back
// exchange, the per-bin diagonal row-sum merge, and the per-bin remote-Kappa
// fetch — posts ALL its messages in one nonblocking wave and completes them with
// a single Waitall: one round-trip that lets unrelated pairs proceed concurrently
// and does not serialize with the neighbour count. The iter index is folded into
// the tag so multi-iter plans cannot mismatch.

// Build dcr_msgs/dcr_reqs from the cached per-iter plan. Requires the flatten
// bases (dcr_iter_imp_base/dcr_iter_exp_base) to be filled already.
inline void cr_dev_build_halo_msgs()
{
    int cap = 0;
    for (int it = 0; it < CR_CSR.num_iters; it++)
    {
        cr_csr_iter *I = &CR_CSR.iters[it];
        for (int r = 0; r < NTask; r++)
            if (I->Recv_count[r] > 0 || I->Send_count[r] > 0)
                cap++;
    }
    dcr_msgs = (cr_halo_msg *)malloc((cap > 0 ? cap : 1) * sizeof(cr_halo_msg));
    dcr_reqs = (MPI_Request *)malloc((cap > 0 ? 2 * cap : 1) * sizeof(MPI_Request));
    ENDRUN_IF(!dcr_msgs || !dcr_reqs, "CR device-solve: halo message list allocation failed");
    dcr_nmsgs = 0;
    for (int it = 0; it < CR_CSR.num_iters; it++)
    {
        cr_csr_iter *I = &CR_CSR.iters[it];
        for (int r = 0; r < NTask; r++)
        {
            if (I->Recv_count[r] == 0 && I->Send_count[r] == 0)
                continue;
            cr_halo_msg *M = &dcr_msgs[dcr_nmsgs++];
            M->rank = r;
            M->it = it;
            M->imp_off = dcr_iter_imp_base[it] + I->Recv_offset[r];
            M->imp_cnt = I->Recv_count[r];
            M->exp_off = dcr_iter_exp_base[it] + I->Send_offset[r];
            M->exp_cnt = I->Send_count[r];
        }
    }
}

inline void cr_dev_free_halo_msgs()
{
    free(dcr_msgs);
    free(dcr_reqs);
    dcr_msgs = NULL;
    dcr_reqs = NULL;
    dcr_nmsgs = 0;
}

// Post the result-back halo (`stride` doubles per import row → owner's export
// slots) as a nonblocking wave. `sendbuf`/`recvbuf` are host buffers; stage them
// with cr_stage_from_device / cr_stage_to_device around the call. stride = 1 for
// the diagonal-merge halo, 2 for the interleaved 2-RHS SpMV. Returns the number
// of requests; complete with cr_dev_halo_wait. Posting and completion are split
// so the caller can run kernels that don't depend on the halo in between.
inline int cr_dev_halo_post_resultback(double *sendbuf, double *recvbuf, int stride)
{
    int n = 0;
    for (int m = 0; m < dcr_nmsgs; m++)
    {
        const cr_halo_msg *M = &dcr_msgs[m];
        if (M->exp_cnt > 0)
            MPI_Irecv(recvbuf + (size_t)stride * M->exp_off, stride * M->exp_cnt, MPI_DOUBLE, M->rank,
                      TAG_CR_HALO_R + M->it, MYMPI_COMM_WORLD, &dcr_reqs[n++]);
        if (M->imp_cnt > 0)
            MPI_Isend(sendbuf + (size_t)stride * M->imp_off, stride * M->imp_cnt, MPI_DOUBLE, M->rank,
                      TAG_CR_HALO_R + M->it, MYMPI_COMM_WORLD, &dcr_reqs[n++]);
    }
    return n;
}

inline void cr_dev_halo_wait(int nreq)
{
    if (nreq > 0)
        MPI_Waitall(nreq, dcr_reqs, MPI_STATUSES_IGNORE);
}

// ---------------------------------------------------------------------------
// Drop import rows that hold no matrix entry.
//
// The tree walk exports a target once per remote top-level node its search
// sphere overlaps, and the overwhelming majority of those speculative exports
// find no pair on the receiving rank: the import side typically carries ~70x
// more rows than it has nonzeros. An empty row contributes an exact zero to
// every cross-rank reduction — the diagonal row sum, the SpMV partial, and all
// four FCT phases — yet still costs a CSR row, 16 doubles of FCT halo, a slot in
// every staged transfer and its share of every MPI message.
//
// Dropping them is exactly result-neutral: the owner-side merges accumulate
// `+= 0.0` and take max/min against the identities, all of which are exact in
// floating point, so the pruned solve is bit-identical to the unpruned one.
//
// Only the server knows which of its rows are empty, so the mask travels back to
// the owners along the existing message plan and both sides compact from the
// same information, keeping rows and export slots in lockstep.

// Server->owner mask exchange: mirrors cr_dev_halo_post_resultback with a
// one-byte payload per row/slot.
inline int cr_dev_halo_post_mask(char *imp_send, char *exp_recv)
{
    int n = 0;
    for (int m = 0; m < dcr_nmsgs; m++)
    {
        const cr_halo_msg *M = &dcr_msgs[m];
        if (M->exp_cnt > 0)
            MPI_Irecv(exp_recv + M->exp_off, M->exp_cnt, MPI_CHAR, M->rank, TAG_CR_PRUNE + M->it, MYMPI_COMM_WORLD,
                      &dcr_reqs[n++]);
        if (M->imp_cnt > 0)
            MPI_Isend(imp_send + M->imp_off, M->imp_cnt, MPI_CHAR, M->rank, TAG_CR_PRUNE + M->it, MYMPI_COMM_WORLD,
                      &dcr_reqs[n++]);
    }
    return n;
}

// Compact dcr_imp_rowptr / dcr_exp_place and the message plan onto the rows that
// carry entries. Must run after cr_dev_build_halo_msgs() (it uses the plan) and
// before anything is sized from dcr_imp_rows / dcr_exp_n. dcr_imp_nnz and the
// column/geometry arrays are untouched: dropping empty rows removes no entry.
inline void cr_dev_prune_empty_import_rows()
{
    const int old_rows = dcr_imp_rows, old_exp = dcr_exp_n;
    if (old_rows <= 0 && old_exp <= 0)
        return;

    char *keep_imp = (char *)malloc((size_t)(old_rows > 0 ? old_rows : 1));
    char *keep_exp = (char *)calloc((size_t)(old_exp > 0 ? old_exp : 1), 1);
    ENDRUN_IF(!keep_imp || !keep_exp, "CR device-solve: import-row prune mask allocation failed");

    long long kept_imp = 0;
    for (int k = 0; k < old_rows; k++)
    {
        keep_imp[k] = (dcr_imp_rowptr[k + 1] > dcr_imp_rowptr[k]) ? 1 : 0;
        kept_imp += keep_imp[k];
    }

    const int nreq = cr_dev_halo_post_mask(keep_imp, keep_exp);
    cr_dev_halo_wait(nreq);

    // New per-message counts and offsets. The flattened arrays are ordered by
    // (comm-loop iter, rank) and so is the message list, so compacting in
    // increasing index order keeps every message's block contiguous and in
    // place; the check below is what guarantees that assumption still holds.
    int *new_ic = (int *)malloc((size_t)(dcr_nmsgs > 0 ? dcr_nmsgs : 1) * sizeof(int));
    int *new_ec = (int *)malloc((size_t)(dcr_nmsgs > 0 ? dcr_nmsgs : 1) * sizeof(int));
    ENDRUN_IF(!new_ic || !new_ec, "CR device-solve: import-row prune plan allocation failed");
    long long covered_imp = 0, covered_exp = 0;
    int prev_iend = 0, prev_eend = 0;
    for (int m = 0; m < dcr_nmsgs; m++)
    {
        const cr_halo_msg *M = &dcr_msgs[m];
        ENDRUN_IF(M->imp_off < prev_iend || M->exp_off < prev_eend,
                  "CR device-solve: halo message %d is out of flat order (imp_off=%d prev_end=%d, exp_off=%d "
                  "prev_end=%d) - prune compaction would scramble the blocks",
                  m, M->imp_off, prev_iend, M->exp_off, prev_eend);
        prev_iend = M->imp_off + M->imp_cnt;
        prev_eend = M->exp_off + M->exp_cnt;
        int ic = 0, ec = 0;
        for (int k = 0; k < M->imp_cnt; k++)
            ic += keep_imp[M->imp_off + k];
        for (int s = 0; s < M->exp_cnt; s++)
            ec += keep_exp[M->exp_off + s];
        new_ic[m] = ic;
        new_ec[m] = ec;
        covered_imp += ic;
        covered_exp += ec;
    }

    // Every row and slot must belong to exactly one message, or the two sides
    // would compact to different lengths and the plan would desynchronise.
    long long kept_exp = 0;
    for (int s = 0; s < old_exp; s++)
        kept_exp += keep_exp[s];
    ENDRUN_IF(covered_imp != kept_imp || covered_exp != kept_exp,
              "CR device-solve: prune mask not covered by the message plan (imp %lld/%lld, exp %lld/%lld)", covered_imp,
              kept_imp, covered_exp, kept_exp);

    // Compact in place; the write cursor never passes the read cursor.
    const int nnz_total = dcr_imp_rowptr[old_rows];
    int j = 0;
    for (int k = 0; k < old_rows; k++)
    {
        const int rs = dcr_imp_rowptr[k];
        if (dcr_imp_rowptr[k + 1] > rs)
            dcr_imp_rowptr[j++] = rs;
    }
    dcr_imp_rowptr[j] = nnz_total;
    dcr_imp_rows = j;

    j = 0;
    for (int s = 0; s < old_exp; s++)
        if (keep_exp[s])
            dcr_exp_place[j++] = dcr_exp_place[s];
    dcr_exp_n = j;

    int ioff = 0, eoff = 0;
    for (int m = 0; m < dcr_nmsgs; m++)
    {
        cr_halo_msg *M = &dcr_msgs[m];
        M->imp_off = ioff;
        M->imp_cnt = new_ic[m];
        M->exp_off = eoff;
        M->exp_cnt = new_ec[m];
        ioff += new_ic[m];
        eoff += new_ec[m];
    }
    ENDRUN_IF(ioff != dcr_imp_rows || eoff != dcr_exp_n,
              "CR device-solve: pruned plan length disagrees with the compacted arrays (%d vs %d rows, %d vs %d slots)",
              ioff, dcr_imp_rows, eoff, dcr_exp_n);

    free(new_ic);
    free(new_ec);
    free(keep_imp);
    free(keep_exp);
}

// Bring the solution back to the host before the per-bin write-back to SphP.
inline void cr_diff_device_solve_download_quantity()
{
#if defined(ACC_OPENACC)
#pragma acc update host(Quantity [0:N_gas])
#elif defined(ACC_OPENMP)
#pragma omp target update from(Quantity [0:N_gas])
#endif // ACC_NOACC: Quantity already lives on the host
}

// ============================================================================
// Geometry-cache device path (CR_DIFF_DEVICE_SOLVE).
//
// The bin-independent geometry (G_iso, G_B) and the full sparsity + halo plan
// are uploaded ONCE per timestep; every per-bin matrix is then recombined ON THE
// DEVICE as w = κ̄(K_i,K_j)·G_iso + κ̄_B(KB_i,KB_j)·G_B together with its
// diagonal. Per bin the only host→device traffic is Kappa[/KappaB] (N_gas) and
// the per-import-row remote coefficients — the nnz-sized arrays never cross
// PCIe after the one upload, and there is no per-bin device alloc/free churn.
// ============================================================================

// One-line census of this module's device working set, printed once per
// timestep. The halo dimensions (dcr_imp_rows, dcr_exp_n) are sums over the
// comm-loop iterations and swing by more than an order of magnitude between
// sync points, and with the FCT limiter each import row carries 16 doubles of
// halo — so the footprint and the largest single staged transfer are what decide
// whether a step fits. Reported as the max over ranks; the device footprint also
// names the rank holding it, which is the one that fails first.
inline void cr_diff_device_geom_report()
{
    const size_t lnnz = (size_t)(dcr_lnnz > 0 ? dcr_lnnz : 1);
    const size_t irows = (size_t)(dcr_imp_rows > 0 ? dcr_imp_rows : 1);
    const size_t innz = (size_t)(dcr_imp_nnz > 0 ? dcr_imp_nnz : 1);
    const size_t en = (size_t)(dcr_exp_n > 0 ? dcr_exp_n : 1);
    const size_t ng = (size_t)N_gas;
    const size_t W = sizeof(cr_csr_w_t);
    const size_t D = sizeof(double);

    // Mirrors the enter-data list at the end of cr_diff_device_geom_upload().
    size_t dev = (ng + 1) * sizeof(int) + lnnz * sizeof(int) + lnnz * W; // local CSR + geometry
    dev += (irows + 1) * sizeof(int) + innz * sizeof(int) + innz * W; // import CSR + geometry
    dev += en * sizeof(int); // dcr_exp_place
    dev += lnnz * W + innz * W; // dcr_w, dcr_imp_w (recombine targets)
    dev += CR_NK * irows * sizeof(MyFloat); // dcr_imp_kkb
    dev += irows * D + en * D; // dcr_imp_out, dcr_exp_in
    dev += 2 * irows * D + 2 * en * D; // dcr_imp_out2, dcr_exp_in2
    dev += ng * D + ng * D; // diag, Quantity
    dev += CR_NK * ng * sizeof(MyFloat); // Kappa [+ KappaB]
    dev += 7 * 2 * ng * D + 2 * ng * sizeof(float); // Quantity2..TVec2 (QuantityOld2 is float)
#ifdef CR_DIFF_ANISO
    dev += lnnz * W + innz * W; // dcr_g_b, dcr_imp_g_b
#endif

    // Largest single host<->device staging request, i.e. what cr_stage_* has to
    // break up. The 8-wide FCT halo dominates everything else by a factor 4.
    size_t xfer = CR_NK * irows * sizeof(MyFloat);

    double loc[7] = {(double)dcr_lnnz, (double)dcr_imp_nnz,      (double)dcr_imp_rows, (double)dcr_exp_n,
                     (double)xfer,     (double)dcr_imp_rows_raw, (double)dcr_exp_n_raw};
    double mx[7];
    MPI_Reduce(loc, mx, 7, MPI_DOUBLE, MPI_MAX, 0, MYMPI_COMM_WORLD);

    struct
    {
        double val;
        int rank;
    } hot = {(double)dev, ThisTask}, hotmax;
    MPI_Reduce(&hot, &hotmax, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MYMPI_COMM_WORLD);

    if (ThisTask == 0)
        printf("CR Diffusion device working set (max over ranks): lnnz=%.0f imp_nnz=%.0f imp_rows=%.0f (raw %.0f, "
               "%.1fx pruned) exp_n=%.0f (raw %.0f) | largest staged transfer=%.3f GB | footprint=%.3f GB on task %d\n",
               mx[0], mx[1], mx[2], mx[5], mx[2] > 0.0 ? mx[5] / mx[2] : 0.0, mx[3], mx[6], mx[4] / 1073741824.0,
               hotmax.val / 1073741824.0, hotmax.rank);
}

// Once per timestep, right after cr_geom_build(): flatten the import-side
// geometry, build the halo-message list, and make EVERYTHING device-resident —
// geometry (copyin), recombine targets dcr_w/dcr_imp_w/diag (alloc;
// device-computed), the Kappa staging, the halo scratch, and all solver vectors.
inline void cr_diff_device_geom_upload()
{
    dcr_row = CR_CSR.row_ptr;
    dcr_col = CR_CSR.col_idx;
    dcr_g_iso = CR_CSR.g_iso;
    dcr_g_b = CR_CSR.g_b;
    dcr_lnnz = CR_CSR.row_ptr[N_gas];

    // ---- size + flatten the per-iter import geometry / export scatter map ----
    // The three totals are sums over the comm-loop iterations and reach ~4e7 on a
    // full-active step, so they are accumulated in 64 bit and range-checked before
    // being narrowed: a wrapped count silently undersizes every flatten buffer
    // below and every device mapping built from it.
    long long imp_rows_ll = 0, imp_nnz_ll = 0, exp_n_ll = 0;
    for (int it = 0; it < CR_CSR.num_iters; it++)
    {
        cr_csr_iter *I = &CR_CSR.iters[it];
        imp_rows_ll += I->Nimport;
        imp_nnz_ll += I->import_row_ptr[I->Nimport];
        exp_n_ll += I->Nexport;
    }
    ENDRUN_IF(imp_rows_ll > 2147483647LL || imp_nnz_ll > 2147483647LL || exp_n_ll > 2147483647LL || dcr_lnnz < 0,
              "CR device-solve: CSR/halo size exceeds 32 bit (lnnz=%d imp_rows=%lld imp_nnz=%lld exp_n=%lld)", dcr_lnnz,
              imp_rows_ll, imp_nnz_ll, exp_n_ll);
    dcr_imp_rows = (int)imp_rows_ll;
    dcr_imp_nnz = (int)imp_nnz_ll;
    dcr_exp_n = (int)exp_n_ll;
    dcr_imp_rows_raw = dcr_imp_rows;
    dcr_exp_n_raw = dcr_exp_n;
    const int nit = CR_CSR.num_iters;
    const int lnnz = dcr_lnnz > 0 ? dcr_lnnz : 1;
    const int innz = dcr_imp_nnz > 0 ? dcr_imp_nnz : 1;
    // Row and slot counts as the tree walk produced them. Only the arrays the
    // flatten itself writes are sized from these; everything else waits until the
    // empty rows have been pruned out below, which typically shrinks the halo by
    // more than an order of magnitude.
    const int raw_irows = dcr_imp_rows > 0 ? dcr_imp_rows : 1;
    const int raw_en = dcr_exp_n > 0 ? dcr_exp_n : 1;

    dcr_iter_imp_base = (int *)malloc((nit > 0 ? nit : 1) * sizeof(int));
    dcr_iter_exp_base = (int *)malloc((nit > 0 ? nit : 1) * sizeof(int));
    dcr_imp_rowptr = (int *)malloc(((size_t)raw_irows + 1) * sizeof(int));
    dcr_imp_col = (int *)malloc((size_t)innz * sizeof(int));
    dcr_imp_g_iso = (cr_csr_w_t *)malloc((size_t)innz * sizeof(cr_csr_w_t));
#ifdef CR_DIFF_ANISO
    dcr_imp_g_b = (cr_csr_w_t *)malloc((size_t)innz * sizeof(cr_csr_w_t));
    ENDRUN_IF(!dcr_imp_g_b, "CR device-solve geometry flatten allocation failed (g_b)");
#endif
    dcr_exp_place = (int *)malloc((size_t)raw_en * sizeof(int));
    ENDRUN_IF(!dcr_iter_imp_base || !dcr_iter_exp_base || !dcr_imp_rowptr || !dcr_imp_col || !dcr_imp_g_iso ||
                  !dcr_exp_place,
              "CR device-solve geometry flatten allocation failed");

    int grow = 0, gnnz = 0, gexp = 0;
    for (int it = 0; it < nit; it++)
    {
        cr_csr_iter *I = &CR_CSR.iters[it];
        dcr_iter_imp_base[it] = grow;
        dcr_iter_exp_base[it] = gexp;
        for (int k = 0; k < I->Nimport; k++)
            dcr_imp_rowptr[grow + k] = gnnz + I->import_row_ptr[k];
        const int inz = I->import_row_ptr[I->Nimport];
        for (int k = 0; k < inz; k++)
        {
            dcr_imp_col[gnnz + k] = I->import_col_idx[k];
            dcr_imp_g_iso[gnnz + k] = I->import_g_iso[k];
#ifdef CR_DIFF_ANISO
            dcr_imp_g_b[gnnz + k] = I->import_g_b[k];
#endif
        }
        for (int s = 0; s < I->Nexport; s++)
            dcr_exp_place[gexp + s] = I->export_place[s];
        grow += I->Nimport;
        gnnz += inz;
        gexp += I->Nexport;
    }
    dcr_imp_rowptr[dcr_imp_rows] = gnnz;

    cr_dev_build_halo_msgs();
    cr_dev_prune_empty_import_rows();

    // Everything from here on is sized from the PRUNED counts.
    const int irows = dcr_imp_rows > 0 ? dcr_imp_rows : 1;
    const int en = dcr_exp_n > 0 ? dcr_exp_n : 1;

    // dcr_w / dcr_imp_w are written by the device recombine; their host backing
    // is only ever touched in ACC_NOACC builds (where it IS the storage), so on
    // GPU builds these pages stay untouched and cost no resident host memory.
    dcr_w = (cr_csr_w_t *)malloc((size_t)lnnz * sizeof(cr_csr_w_t));
    dcr_imp_w = (cr_csr_w_t *)malloc((size_t)innz * sizeof(cr_csr_w_t));
    dcr_imp_out = (double *)malloc((size_t)irows * sizeof(double));
    dcr_exp_in = (double *)malloc((size_t)en * sizeof(double));
    dcr_imp_kkb = (MyFloat *)malloc((size_t)CR_NK * irows * sizeof(MyFloat));
    dcr_exp_kkb = (MyFloat *)malloc((size_t)CR_NK * en * sizeof(MyFloat));
    ENDRUN_IF(!dcr_w || !dcr_imp_w || !dcr_imp_out || !dcr_exp_in || !dcr_imp_kkb || !dcr_exp_kkb,
              "CR device-solve halo scratch allocation failed");
    dcr_imp_out2 = (double *)malloc(2 * (size_t)irows * sizeof(double));
    dcr_exp_in2 = (double *)malloc(2 * (size_t)en * sizeof(double));
    ENDRUN_IF(!dcr_imp_out2 || !dcr_exp_in2, "CR device-solve 2-RHS halo scratch allocation failed");


#if defined(ACC_OPENACC)
#pragma acc enter data copyin(dcr_row [0:N_gas + 1], dcr_col [0:lnnz], dcr_g_iso [0:lnnz])
#ifdef CR_DIFF_ANISO
#pragma acc enter data copyin(dcr_g_b [0:lnnz], dcr_imp_g_b [0:innz])
#endif
#pragma acc enter data copyin(dcr_imp_rowptr [0:irows + 1], dcr_imp_col [0:innz], dcr_imp_g_iso [0:innz],              \
                              dcr_exp_place [0:en])
#pragma acc enter data create(dcr_w [0:lnnz], dcr_imp_w [0:innz], dcr_imp_kkb [0:CR_NK * irows])
#pragma acc enter data create(dcr_imp_out [0:irows], dcr_exp_in [0:en])
#pragma acc enter data create(diag [0:N_gas], KAPPAB_ZERO_NGAS Kappa [0:N_gas])
#pragma acc enter data create(Quantity [0:N_gas])
#pragma acc enter data create(dcr_imp_out2 [0:2 * irows], dcr_exp_in2 [0:2 * en])
#pragma acc enter data create(Quantity2 [0:2 * N_gas], QuantityOld2 [0:2 * N_gas], Residual2 [0:2 * N_gas],            \
                              DVec2 [0:2 * N_gas], PVec2 [0:2 * N_gas], QVec2 [0:2 * N_gas], SVec2 [0:2 * N_gas],      \
                              TVec2 [0:2 * N_gas])
#elif defined(ACC_OPENMP)
#pragma omp target enter data map(to : dcr_row [0:N_gas + 1], dcr_col [0:lnnz], dcr_g_iso [0:lnnz])
#ifdef CR_DIFF_ANISO
#pragma omp target enter data map(to : dcr_g_b [0:lnnz], dcr_imp_g_b [0:innz])
#endif
#pragma omp target enter data map(to                                                                                   \
                                  : dcr_imp_rowptr [0:irows + 1], dcr_imp_col [0:innz], dcr_imp_g_iso [0:innz],        \
                                    dcr_exp_place [0:en])
#pragma omp target enter data map(alloc : dcr_w [0:lnnz], dcr_imp_w [0:innz], dcr_imp_kkb [0:CR_NK * irows])
#pragma omp target enter data map(alloc : dcr_imp_out [0:irows], dcr_exp_in [0:en])
#pragma omp target enter data map(alloc : diag [0:N_gas], KAPPAB_ZERO_NGAS Kappa [0:N_gas])
#pragma omp target enter data map(alloc : Quantity [0:N_gas])
#pragma omp target enter data map(alloc : dcr_imp_out2 [0:2 * irows], dcr_exp_in2 [0:2 * en])
#pragma omp target enter data map(alloc                                                                                \
                                  : Quantity2 [0:2 * N_gas], QuantityOld2 [0:2 * N_gas], Residual2 [0:2 * N_gas],      \
                                    DVec2 [0:2 * N_gas], PVec2 [0:2 * N_gas], QVec2 [0:2 * N_gas],                     \
                                    SVec2 [0:2 * N_gas], TVec2 [0:2 * N_gas])
#endif // ACC_OPENACC / ACC_OPENMP  (ACC_NOACC: arrays stay on the host)

    cr_diff_device_geom_report();
}

// Per-bin matrix recombine, all on the device. Kappa[]/KappaB[] must hold the
// current bin's coefficients (Kappa loop) on entry. Overlap structure: the
// remote-Kappa fetch is in flight while the LOCAL rows recombine; the diagonal
// row-sum merge halo is the only exposed exchange.
inline void cr_diff_device_bin_recombine()
{
    TIMER_PUSH(1, "CR_GEOM_RECOMBINE");

    // ---- this bin's coefficients → device ----
#if defined(ACC_OPENACC)
#pragma acc update device(KAPPAB_ZERO_NGAS Kappa [0:N_gas])
#elif defined(ACC_OPENMP)
#pragma omp target update to(KAPPAB_ZERO_NGAS Kappa [0:N_gas])
#endif

    // ---- post the in-get halo: remote targets' Kappa[/KappaB] for our import
    //      rows. Packed stride-CR_NK so both coefficients ride one message. The
    //      buffers are host-side (Kappa is computed on the host); the receive
    //      lands in dcr_imp_kkb and is uploaded once the wave completes. ----
    int nreq_k = 0;
    if (dcr_imp_rows > 0 || dcr_exp_n > 0)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int s = 0; s < dcr_exp_n; s++)
        {
            const int place = dcr_exp_place[s];
            dcr_exp_kkb[CR_NK * s] = Kappa[place];
#ifdef CR_DIFF_ANISO
            dcr_exp_kkb[CR_NK * s + 1] = KappaB[place];
#endif
        }
        for (int m = 0; m < dcr_nmsgs; m++)
        {
            const cr_halo_msg *M = &dcr_msgs[m];
            if (M->imp_cnt > 0)
                MPI_Irecv(dcr_imp_kkb + (size_t)CR_NK * M->imp_off, CR_NK * M->imp_cnt, MPI_MyFloat, M->rank,
                          TAG_CR_HALO_K + M->it, MYMPI_COMM_WORLD, &dcr_reqs[nreq_k++]);
            if (M->exp_cnt > 0)
                MPI_Isend(dcr_exp_kkb + (size_t)CR_NK * M->exp_off, CR_NK * M->exp_cnt, MPI_MyFloat, M->rank,
                          TAG_CR_HALO_K + M->it, MYMPI_COMM_WORLD, &dcr_reqs[nreq_k++]);
        }
    }

    // ---- local rows (overlaps with the in-flight Kappa halo):
    //      w_k = κ̄(K_i,K_j)·G_iso_k + κ̄_B(KB_i,KB_j)·G_B_k,  diag_i = 1 + Σ_k w_k
    //      Same >0 guards as the matrix-free evaluator's harmonic means. ----
#if defined(ACC_OPENACC)
#ifdef CR_DIFF_ANISO
#pragma acc parallel loop present(Kappa [0:N_gas], KappaB [0:N_gas], diag [0:N_gas], cr_active [0:N_gas],              \
                                  dcr_row [0:N_gas + 1], dcr_col [0:dcr_lnnz], dcr_g_iso [0:dcr_lnnz],                 \
                                  dcr_g_b [0:dcr_lnnz], FCT_WAD_PRESENT dcr_w [0:dcr_lnnz])
#else
#pragma acc parallel loop present(Kappa [0:N_gas], diag [0:N_gas], cr_active [0:N_gas], dcr_row [0:N_gas + 1],         \
                                  dcr_col [0:dcr_lnnz], dcr_g_iso [0:dcr_lnnz], FCT_WAD_PRESENT dcr_w [0:dcr_lnnz])
#endif
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for
#elif defined(ACC_NOACC)
#pragma omp parallel for
#endif
    for (int i = 0; i < N_gas; i++)
    {
        if (cr_active[i])
        {
            const double Ki = Kappa[i];
#ifdef CR_DIFF_ANISO
            const double Kbi = KappaB[i];
#endif
            double s = 1.0;
            const int rs = dcr_row[i];
            const int re = dcr_row[i + 1];
            for (int k = rs; k < re; k++)
            {
                const int jc = dcr_col[k];
                const double Kj = Kappa[jc];
                double w = (Ki > 0.0 && Kj > 0.0) ? (2.0 * Ki * Kj / (Ki + Kj)) * (double)dcr_g_iso[k] : 0.0;
#ifdef CR_DIFF_ANISO
                const double Kbj = KappaB[jc];
                if (Kbi + Kbj > 0.0)
                    w += (2.0 * Kbi * Kbj / (Kbi + Kbj)) * (double)dcr_g_b[k];
#endif
                dcr_w[k] = (cr_csr_w_t)w;
                s += w;
            }
            diag[i] = s;
        }
        else
        {
            diag[i] = 0.0;
        }
    }

    if (dcr_imp_rows > 0 || dcr_exp_n > 0)
    {
        // ---- complete the Kappa halo and stage it onto the device ----
        cr_dev_halo_wait(nreq_k);
        if (dcr_imp_rows > 0)
            cr_stage_to_device(dcr_imp_kkb, (size_t)CR_NK * dcr_imp_rows);

        // ---- import rows: recombine w and accumulate the row sums ----
        if (dcr_imp_rows > 0)
        {
#if defined(ACC_OPENACC)
#ifdef CR_DIFF_ANISO
#pragma acc parallel loop present(Kappa [0:N_gas], KappaB [0:N_gas], dcr_imp_rowptr [0:dcr_imp_rows + 1],              \
                                  dcr_imp_col [0:dcr_imp_nnz], dcr_imp_g_iso [0:dcr_imp_nnz],                          \
                                  dcr_imp_g_b [0:dcr_imp_nnz], FCT_IMP_WAD_PRESENT dcr_imp_w [0:dcr_imp_nnz],          \
                                  dcr_imp_kkb [0:CR_NK * dcr_imp_rows], dcr_imp_out [0:dcr_imp_rows])
#else
#pragma acc parallel loop present(Kappa [0:N_gas], dcr_imp_rowptr [0:dcr_imp_rows + 1], dcr_imp_col [0:dcr_imp_nnz],   \
                                  dcr_imp_g_iso [0:dcr_imp_nnz], FCT_IMP_WAD_PRESENT dcr_imp_w [0:dcr_imp_nnz],        \
                                  dcr_imp_kkb [0:CR_NK * dcr_imp_rows], dcr_imp_out [0:dcr_imp_rows])
#endif
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for
#elif defined(ACC_NOACC)
#pragma omp parallel for
#endif
            for (int g = 0; g < dcr_imp_rows; g++)
            {
                const double Kg = (double)dcr_imp_kkb[CR_NK * g];
#ifdef CR_DIFF_ANISO
                const double Kbg = (double)dcr_imp_kkb[CR_NK * g + 1];
#endif
                double s = 0.0;
                const int rs = dcr_imp_rowptr[g];
                const int re = dcr_imp_rowptr[g + 1];
                for (int k = rs; k < re; k++)
                {
                    const int jc = dcr_imp_col[k];
                    const double Kj = Kappa[jc];
                    double w = (Kg > 0.0 && Kj > 0.0) ? (2.0 * Kg * Kj / (Kg + Kj)) * (double)dcr_imp_g_iso[k] : 0.0;
#ifdef CR_DIFF_ANISO
                    const double Kbj = KappaB[jc];
                    if (Kbg + Kbj > 0.0)
                        w += (2.0 * Kbg * Kbj / (Kbg + Kbj)) * (double)dcr_imp_g_b[k];
#endif
                    dcr_imp_w[k] = (cr_csr_w_t)w;
                    s += w;
                }
                dcr_imp_out[g] = s; // row sum, merged into the owner's diagonal below
            }
        }

        // ---- result-back halo of the import row sums; merge into diag ----
        // Exchanged on the HOST like every other halo in this module: MPICH here
        // cannot take a device allocation of this size and fails the send outright
        // ("Fatal error in internal_Isend").
        if (dcr_imp_rows > 0)
            cr_stage_from_device(dcr_imp_out, (size_t)dcr_imp_rows);
        const int nreq_d = cr_dev_halo_post_resultback(dcr_imp_out, dcr_exp_in, 1);
        cr_dev_halo_wait(nreq_d);
        if (dcr_exp_n > 0)
            cr_stage_to_device(dcr_exp_in, (size_t)dcr_exp_n);

        if (dcr_exp_n > 0)
        {
#if defined(ACC_OPENACC)
#pragma acc parallel loop present(diag [0:N_gas], dcr_exp_in [0:dcr_exp_n], dcr_exp_place [0:dcr_exp_n])
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for
#elif defined(ACC_NOACC)
#pragma omp parallel for
#endif
            for (int s = 0; s < dcr_exp_n; s++)
            {
                const int place = dcr_exp_place[s];
                const double v = dcr_exp_in[s];
                // same `place` may receive from several ranks → atomic
#if defined(ACC_OPENACC)
#pragma acc atomic update
#elif defined(ACC_OPENMP) || defined(ACC_NOACC)
#pragma omp atomic update
#endif
                diag[place] += v;
            }
        }
    }

    TIMER_POP(1, "CR_GEOM_RECOMBINE");
}

// Once per timestep, before cr_csr_free(): release the device residency and the
// flatten buffers. dcr_row/dcr_col/dcr_g_iso/dcr_g_b alias CR_CSR and are freed
// by cr_csr_free afterwards — the exit-data pragmas only look the (still valid)
// host pointers up.
inline void cr_diff_device_geom_teardown()
{
    const int lnnz = dcr_lnnz > 0 ? dcr_lnnz : 1;
    const int irows = dcr_imp_rows > 0 ? dcr_imp_rows : 1;
    const int innz = dcr_imp_nnz > 0 ? dcr_imp_nnz : 1;
    const int en = dcr_exp_n > 0 ? dcr_exp_n : 1;
#if defined(ACC_OPENACC)
#pragma acc exit data delete (diag [0:N_gas], Quantity [0:N_gas])
#pragma acc exit data delete (dcr_imp_out2 [0:2 * irows], dcr_exp_in2 [0:2 * en])
#pragma acc exit data delete (Quantity2 [0:2 * N_gas], QuantityOld2 [0:2 * N_gas], Residual2 [0:2 * N_gas],            \
                              DVec2 [0:2 * N_gas], PVec2 [0:2 * N_gas], QVec2 [0:2 * N_gas], SVec2 [0:2 * N_gas],      \
                              TVec2 [0:2 * N_gas])
#pragma acc exit data delete (KAPPAB_ZERO_NGAS Kappa [0:N_gas])
#pragma acc exit data delete (dcr_imp_out [0:irows], dcr_exp_in [0:en], dcr_imp_kkb [0:CR_NK * irows])
#pragma acc exit data delete (dcr_w [0:lnnz], dcr_imp_w [0:innz])
#pragma acc exit data delete (dcr_imp_rowptr [0:irows + 1], dcr_imp_col [0:innz], dcr_imp_g_iso [0:innz],              \
                              dcr_exp_place [0:en])
#ifdef CR_DIFF_ANISO
#pragma acc exit data delete (dcr_g_b [0:lnnz], dcr_imp_g_b [0:innz])
#endif
#pragma acc exit data delete (dcr_row [0:N_gas + 1], dcr_col [0:lnnz], dcr_g_iso [0:lnnz])
#elif defined(ACC_OPENMP)
#pragma omp target exit data map(delete : diag [0:N_gas], Quantity [0:N_gas])
#pragma omp target exit data map(delete : dcr_imp_out2 [0:2 * irows], dcr_exp_in2 [0:2 * en])
#pragma omp target exit data map(delete                                                                                \
                                 : Quantity2 [0:2 * N_gas], QuantityOld2 [0:2 * N_gas], Residual2 [0:2 * N_gas],       \
                                   DVec2 [0:2 * N_gas], PVec2 [0:2 * N_gas], QVec2 [0:2 * N_gas], SVec2 [0:2 * N_gas], \
                                   TVec2 [0:2 * N_gas])
#pragma omp target exit data map(delete : KAPPAB_ZERO_NGAS Kappa [0:N_gas])
#pragma omp target exit data map(delete : dcr_imp_out [0:irows], dcr_exp_in [0:en], dcr_imp_kkb [0:CR_NK * irows])
#pragma omp target exit data map(delete : dcr_w [0:lnnz], dcr_imp_w [0:innz])
#pragma omp target exit data map(delete                                                                                \
                                 : dcr_imp_rowptr [0:irows + 1], dcr_imp_col [0:innz], dcr_imp_g_iso [0:innz],         \
                                   dcr_exp_place [0:en])
#ifdef CR_DIFF_ANISO
#pragma omp target exit data map(delete : dcr_g_b [0:lnnz], dcr_imp_g_b [0:innz])
#endif
#pragma omp target exit data map(delete : dcr_row [0:N_gas + 1], dcr_col [0:lnnz], dcr_g_iso [0:lnnz])
#endif // ACC_OPENACC / ACC_OPENMP

    cr_dev_free_halo_msgs();
    free(dcr_iter_imp_base);
    free(dcr_iter_exp_base);
    free(dcr_imp_rowptr);
    free(dcr_imp_col);
    free(dcr_imp_g_iso);
#ifdef CR_DIFF_ANISO
    free(dcr_imp_g_b);
    dcr_imp_g_b = NULL;
#endif
    free(dcr_imp_out2);
    free(dcr_exp_in2);
    dcr_imp_out2 = dcr_exp_in2 = NULL;
    free(dcr_w);
    free(dcr_imp_w);
    free(dcr_exp_place);
    free(dcr_imp_out);
    free(dcr_exp_in);
    free(dcr_imp_kkb);
    free(dcr_exp_kkb);
    dcr_iter_imp_base = dcr_iter_exp_base = NULL;
    dcr_imp_rowptr = dcr_imp_col = NULL;
    dcr_imp_g_iso = NULL;
    dcr_w = dcr_imp_w = NULL;
    dcr_exp_place = NULL;
    dcr_imp_out = dcr_exp_in = NULL;
    dcr_imp_kkb = dcr_exp_kkb = NULL;
    dcr_row = dcr_col = NULL;
    dcr_g_iso = dcr_g_b = NULL;
    dcr_lnnz = dcr_imp_rows = dcr_imp_nnz = dcr_exp_n = 0;
}

// ============================================================================
// 2-RHS fused solve (CR_DIFF_DEVICE_SOLVE).
//
// Energy (rhs 0) and number (rhs 1) of a component share the matrix A, so both
// right-hand sides ride one BiCGStab with independent scalar recurrences. All
// vectors are INTERLEAVED, vec2[2*i + rhs]: the SpMV's gather of a neighbour's
// two values is one adjacent 16-byte load, and the (dominant) matrix stream —
// dcr_w/dcr_col — is read ONCE for two solves. The halo ships 2 doubles per
// import row through the same message plan (stride 2). Per-RHS scalars arrive
// as separate arguments; dead lanes are gated via live0/live1.
// ============================================================================

// Refresh the device copy of both interleaved right-hand sides (Quantity2 =
// b = x0 for rhs 0/1, QuantityOld2 = convergence floors). Called once per bin.
inline void cr_diff_device_rhs2_upload()
{
#if defined(ACC_OPENACC)
#pragma acc update device(Quantity2 [0:2 * N_gas], QuantityOld2 [0:2 * N_gas])
#elif defined(ACC_OPENMP)
#pragma omp target update to(Quantity2 [0:2 * N_gas], QuantityOld2 [0:2 * N_gas])
#endif // ACC_NOACC: arrays already on the host
}

// out2 = A * in2 for both interleaved RHS in one matrix pass. Structure matches
// cr_diffusion_csr_spmv_gpu: import partial sums first, halo posted (stride 2),
// local rows overlap the in-flight halo, then wait + atomic scatter.
inline void cr_diffusion_csr_spmv_gpu2(double *in2, double *out2)
{
    int nreq = 0;

    // ---- off-rank partial sums for the remote rows we were asked to serve ----
    if (dcr_imp_rows > 0)
    {
#if defined(ACC_OPENACC)
#pragma acc parallel loop present(in2 [0:2 * N_gas], dcr_imp_rowptr [0:dcr_imp_rows + 1], dcr_imp_col [0:dcr_imp_nnz], \
                                  dcr_imp_w [0:dcr_imp_nnz], dcr_imp_out2 [0:2 * dcr_imp_rows])
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for
#elif defined(ACC_NOACC)
#pragma omp parallel for
#endif
        for (int g = 0; g < dcr_imp_rows; g++)
        {
            double s0 = 0.0, s1 = 0.0;
            const int rs = dcr_imp_rowptr[g];
            const int re = dcr_imp_rowptr[g + 1];
            for (int k = rs; k < re; k++)
            {
                const double w = dcr_imp_w[k];
                const int jc = dcr_imp_col[k];
                s0 -= w * in2[2 * jc];
                s1 -= w * in2[2 * jc + 1];
            }
            dcr_imp_out2[2 * g] = s0;
            dcr_imp_out2[2 * g + 1] = s1;
        }
    }

    // ---- post the halo (2 doubles per slot) ----
    // Staged through the host like every other halo in this module: MPICH here
    // cannot take a device allocation of this size and fails the send outright
    // ("Fatal error in internal_Isend"). The download precedes the post, so the
    // local-rows kernel below still overlaps the MPI wave.
    if (dcr_imp_rows > 0 || dcr_exp_n > 0)
    {
        if (dcr_imp_rows > 0)
            cr_stage_from_device(dcr_imp_out2, (size_t)2 * dcr_imp_rows);
        nreq = cr_dev_halo_post_resultback(dcr_imp_out2, dcr_exp_in2, 2);
    }

    // ---- local rows + diagonal, both RHS, one matrix read (overlaps halo) ----
#if defined(ACC_OPENACC)
#pragma acc parallel loop present(out2 [0:2 * N_gas], in2 [0:2 * N_gas], diag [0:N_gas], cr_active [0:N_gas],          \
                                  dcr_row [0:N_gas + 1], dcr_col [0:dcr_lnnz], dcr_w [0:dcr_lnnz])
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for
#elif defined(ACC_NOACC)
#pragma omp parallel for
#endif
    for (int i = 0; i < N_gas; i++)
    {
        if (cr_active[i])
        {
            const double d = diag[i];
            double s0 = in2[2 * i] * d;
            double s1 = in2[2 * i + 1] * d;
            const int rs = dcr_row[i];
            const int re = dcr_row[i + 1];
            for (int k = rs; k < re; k++)
            {
                const double w = dcr_w[k];
                const int jc = dcr_col[k];
                s0 -= w * in2[2 * jc];
                s1 -= w * in2[2 * jc + 1];
            }
            out2[2 * i] = s0;
            out2[2 * i + 1] = s1;
        }
        else
        {
            out2[2 * i] = out2[2 * i + 1] = 0.0;
        }
    }

    // ---- complete the halo and scatter-add into our local rows ----
    if (nreq > 0 || dcr_exp_n > 0)
    {
        TIMER_PUSH(1, "CR_CSR_SPMV_COMM");
        cr_dev_halo_wait(nreq);
        if (dcr_exp_n > 0)
            cr_stage_to_device(dcr_exp_in2, (size_t)2 * dcr_exp_n);
        TIMER_POP(1, "CR_CSR_SPMV_COMM");

        if (dcr_exp_n > 0)
        {
#if defined(ACC_OPENACC)
#pragma acc parallel loop present(out2 [0:2 * N_gas], dcr_exp_in2 [0:2 * dcr_exp_n], dcr_exp_place [0:dcr_exp_n])
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for
#elif defined(ACC_NOACC)
#pragma omp parallel for
#endif
            for (int s = 0; s < dcr_exp_n; s++)
            {
                const int place = dcr_exp_place[s];
                const double v0 = dcr_exp_in2[2 * s];
                const double v1 = dcr_exp_in2[2 * s + 1];
                // same `place` may receive from several ranks → atomic
#if defined(ACC_OPENACC)
#pragma acc atomic update
#elif defined(ACC_OPENMP) || defined(ACC_NOACC)
#pragma omp atomic update
#endif
                out2[2 * place] += v0;
#if defined(ACC_OPENACC)
#pragma acc atomic update
#elif defined(ACC_OPENMP) || defined(ACC_NOACC)
#pragma omp atomic update
#endif
                out2[2 * place + 1] += v1;
            }
        }
    }
}

// r0 = M⁻¹(b − A x0) per RHS; d = r0; p = q = 0. On entry Residual2 holds A*x0
// (from the SpMV) and Quantity2 holds b = x0.
inline void cr_dev_bicgstab_init2()
{
#if defined(ACC_OPENACC)
#pragma acc parallel loop present(Residual2 [0:2 * N_gas], Quantity2 [0:2 * N_gas], diag [0:N_gas],                    \
                                  DVec2 [0:2 * N_gas], PVec2 [0:2 * N_gas], QVec2 [0:2 * N_gas], SVec2 [0:2 * N_gas],  \
                                  TVec2 [0:2 * N_gas], cr_active [0:N_gas])
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for
#elif defined(ACC_NOACC)
#pragma omp parallel for
#endif
    for (int i = 0; i < N_gas; i++)
    {
        if (cr_active[i])
        {
            const double dinv = 1.0 / diag[i];
            for (int r = 0; r < 2; r++)
            {
                const int ir = 2 * i + r;
                Residual2[ir] = (Quantity2[ir] - Residual2[ir]) * dinv;
                DVec2[ir] = Residual2[ir];
                PVec2[ir] = QVec2[ir] = 0.0;
            }
        }
        else
        {
            for (int r = 0; r < 2; r++)
            {
                const int ir = 2 * i + r;
                Residual2[ir] = DVec2[ir] = QVec2[ir] = 0.0;
                PVec2[ir] = 0.0;
                SVec2[ir] = TVec2[ir] = 0.0;
            }
        }
    }
}

// local Σ a·b per RHS over active particles (caller MPI_Allreduces both)
inline void cr_dev_dot2(double *a2, double *b2, double *loc0, double *loc1)
{
    double l0 = 0.0, l1 = 0.0;
#if defined(ACC_OPENACC)
#pragma acc parallel loop present(a2[0 : 2 * N_gas], b2[0 : 2 * N_gas], cr_active[0 : N_gas]) reduction(+ : l0)        \
    reduction(+ : l1)
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for reduction(+ : l0) reduction(+ : l1)
#elif defined(ACC_NOACC)
#pragma omp parallel for reduction(+ : l0) reduction(+ : l1)
#endif
    for (int i = 0; i < N_gas; i++)
        if (cr_active[i])
        {
            l0 += a2[2 * i] * b2[2 * i];
            l1 += a2[2 * i + 1] * b2[2 * i + 1];
        }
    *loc0 = l0;
    *loc1 = l1;
}

// p_r = r_r + β_r (p_r − γ_r q_r), per RHS. Dead lanes (live_r == 0) are skipped
// entirely rather than relying on β = 0: a dead lane's q may hold the NaN that
// killed it, and 0·NaN would re-poison p.
inline void cr_dev_pvec_update2(double beta0, double beta1, double gamma0, double gamma1, int live0, int live1)
{
#if defined(ACC_OPENACC)
#pragma acc parallel loop present(PVec2 [0:2 * N_gas], Residual2 [0:2 * N_gas], QVec2 [0:2 * N_gas],                   \
                                  cr_active [0:N_gas])
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for
#elif defined(ACC_NOACC)
#pragma omp parallel for
#endif
    for (int i = 0; i < N_gas; i++)
        if (cr_active[i])
        {
            if (live0)
                PVec2[2 * i] = Residual2[2 * i] + beta0 * (PVec2[2 * i] - gamma0 * QVec2[2 * i]);
            if (live1)
                PVec2[2 * i + 1] = Residual2[2 * i + 1] + beta1 * (PVec2[2 * i + 1] - gamma1 * QVec2[2 * i + 1]);
        }
}

// q /= diag (in place, both RHS); return local Σ d·q per RHS
inline void cr_dev_qscale_dot2(double *loc0, double *loc1)
{
    double l0 = 0.0, l1 = 0.0;
#if defined(ACC_OPENACC)
#pragma acc parallel loop present(QVec2[0 : 2 * N_gas], diag[0 : N_gas], DVec2[0 : 2 * N_gas], cr_active[0 : N_gas])   \
    reduction(+ : l0) reduction(+ : l1)
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for reduction(+ : l0) reduction(+ : l1)
#elif defined(ACC_NOACC)
#pragma omp parallel for reduction(+ : l0) reduction(+ : l1)
#endif
    for (int i = 0; i < N_gas; i++)
        if (cr_active[i])
        {
            const double dinv = 1.0 / diag[i];
            QVec2[2 * i] *= dinv;
            QVec2[2 * i + 1] *= dinv;
            l0 += DVec2[2 * i] * QVec2[2 * i];
            l1 += DVec2[2 * i + 1] * QVec2[2 * i + 1];
        }
    *loc0 = l0;
    *loc1 = l1;
}

// s_r = r_r − α_r q_r, per RHS (dead lanes skipped — see cr_dev_pvec_update2)
inline void cr_dev_svec_update2(double alpha0, double alpha1, int live0, int live1)
{
#if defined(ACC_OPENACC)
#pragma acc parallel loop present(SVec2 [0:2 * N_gas], Residual2 [0:2 * N_gas], QVec2 [0:2 * N_gas],                   \
                                  cr_active [0:N_gas])
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for
#elif defined(ACC_NOACC)
#pragma omp parallel for
#endif
    for (int i = 0; i < N_gas; i++)
        if (cr_active[i])
        {
            if (live0)
                SVec2[2 * i] = Residual2[2 * i] - alpha0 * QVec2[2 * i];
            if (live1)
                SVec2[2 * i + 1] = Residual2[2 * i + 1] - alpha1 * QVec2[2 * i + 1];
        }
}

// t /= diag (in place, both RHS); return local (t,s) and (t,t) per RHS
inline void cr_dev_tscale_dots2(double *loc_ts0, double *loc_tt0, double *loc_ts1, double *loc_tt1)
{
    double ts0 = 0.0, tt0 = 0.0, ts1 = 0.0, tt1 = 0.0;
#if defined(ACC_OPENACC)
#pragma acc parallel loop present(TVec2[0 : 2 * N_gas], diag[0 : N_gas], SVec2[0 : 2 * N_gas], cr_active[0 : N_gas])   \
    reduction(+ : ts0) reduction(+ : tt0) reduction(+ : ts1) reduction(+ : tt1)
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for reduction(+ : ts0) reduction(+ : tt0) reduction(+ : ts1)              \
    reduction(+ : tt1)
#elif defined(ACC_NOACC)
#pragma omp parallel for reduction(+ : ts0) reduction(+ : tt0) reduction(+ : ts1) reduction(+ : tt1)
#endif
    for (int i = 0; i < N_gas; i++)
        if (cr_active[i])
        {
            const double dinv = 1.0 / diag[i];
            TVec2[2 * i] *= dinv;
            TVec2[2 * i + 1] *= dinv;
            ts0 += TVec2[2 * i] * SVec2[2 * i];
            tt0 += TVec2[2 * i] * TVec2[2 * i];
            ts1 += TVec2[2 * i + 1] * SVec2[2 * i + 1];
            tt1 += TVec2[2 * i + 1] * TVec2[2 * i + 1];
        }
    *loc_ts0 = ts0;
    *loc_tt0 = tt0;
    *loc_ts1 = ts1;
    *loc_tt1 = tt1;
}

// x_r += α_r p_r + γ_r s_r; r_r = s_r − γ_r t_r; fused per-RHS (d̃0, r_new) [next
// delta1, for the β recurrence] and the residual 2-norm ||r_new||² [convergence].
// Dead lanes (live_r == 0) are skipped entirely so a dead RHS's x and r freeze
// EXACTLY — matching the sequential path's "break before the update" breakdown
// semantics — and so 0·NaN from a poisoned dead lane can never reach the stored
// solution.
inline void cr_dev_solution_update2(double alpha0, double alpha1, double gamma0, double gamma1, int live0, int live1,
                                    double *loc_rr0, double *loc_rr1, double *loc_d10, double *loc_d11)
{
    double rr0 = 0.0, rr1 = 0.0, ld0 = 0.0, ld1 = 0.0;
#if defined(ACC_OPENACC)
#pragma acc parallel loop present(Quantity2[0 : 2 * N_gas], Residual2[0 : 2 * N_gas], PVec2[0 : 2 * N_gas],            \
                                  SVec2[0 : 2 * N_gas], TVec2[0 : 2 * N_gas], DVec2[0 : 2 * N_gas],                    \
                                  cr_active[0 : N_gas]) reduction(+ : rr0) reduction(+ : rr1) reduction(+ : ld0)       \
    reduction(+ : ld1)
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for reduction(+ : rr0) reduction(+ : rr1) reduction(+ : ld0)              \
    reduction(+ : ld1)
#elif defined(ACC_NOACC)
#pragma omp parallel for reduction(+ : rr0) reduction(+ : rr1) reduction(+ : ld0) reduction(+ : ld1)
#endif
    for (int i = 0; i < N_gas; i++)
        if (cr_active[i])
        {
            if (live0)
            {
                const int ir = 2 * i;
                Quantity2[ir] += alpha0 * PVec2[ir] + gamma0 * SVec2[ir];
                const double rnew = SVec2[ir] - gamma0 * TVec2[ir];
                Residual2[ir] = rnew;
                rr0 += rnew * rnew; // ||r_new||² (convergence + conservation control)
                ld0 += DVec2[ir] * rnew; // (d̃0, r_new) = next delta1 (β recurrence)
            }
            if (live1)
            {
                const int ir = 2 * i + 1;
                Quantity2[ir] += alpha1 * PVec2[ir] + gamma1 * SVec2[ir];
                const double rnew = SVec2[ir] - gamma1 * TVec2[ir];
                Residual2[ir] = rnew;
                rr1 += rnew * rnew;
                ld1 += DVec2[ir] * rnew;
            }
        }
    *loc_rr0 = rr0;
    *loc_rr1 = rr1;
    *loc_d10 = ld0;
    *loc_d11 = ld1;
}

// De-interleave one RHS's solution into the 1-wide Quantity[] so the existing
// download + write-back path runs unchanged.
inline void cr_dev_extract_rhs2(int rhs)
{
#if defined(ACC_OPENACC)
#pragma acc parallel loop present(Quantity [0:N_gas], Quantity2 [0:2 * N_gas], cr_active [0:N_gas])
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for
#elif defined(ACC_NOACC)
#pragma omp parallel for
#endif
    for (int i = 0; i < N_gas; i++)
        Quantity[i] = cr_active[i] ? Quantity2[2 * i + rhs] : 0.0;
}

#endif // CR_DIFF_DEVICE_SOLVE
