// Native loop only: preserve the model's solver and complete integration state.
#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdint>
#include <cstring>

extern "C" int door_mujoco_version() { return mj_version(); }

extern "C" int door_batch(const mjModel* m, mjData* d, int count, int skip,
                          int terminal, double* states, int64_t* frames,
                          const double* actions, const double* mid, const double* half,
                          const double* initial_q, const double* initial_v,
                          double* q, double* v, double* next_q, double* next_v,
                          uint8_t* done) {
  if (m->nq != 30 || m->nv != 30 || m->nu != 28) return 1;
  const unsigned int spec = mjSTATE_INTEGRATION;
  const int width = mj_stateSize(m, spec);
  for (int i = 0; i < count; ++i) {
    mj_setState(m, d, states + i*width, spec);
    for (int j = 0; j < m->nv; ++j)
      if (d->qfrc_applied[j] != 0) return 2;
    for (int j = 0; j < 6*m->nbody; ++j)
      if (d->xfrc_applied[j] != 0) return 2;
    for (int j = 0; j < 28; ++j) d->ctrl[j] = mid[j] + actions[i*28+j]*half[j];
    for (int s = 0; s < skip; ++s) {
      mj_step(m, d);
      for (int j = 0; j < 30; ++j)
        if (!std::isfinite(d->qpos[j]) || !std::isfinite(d->qvel[j])) return 3;
    }
    std::memcpy(next_q+i*30, d->qpos, 30*sizeof(double));
    std::memcpy(next_v+i*30, d->qvel, 30*sizeof(double));
    done[i] = ++frames[i] >= terminal;
    if (done[i]) {
      mj_resetData(m, d);
      std::memcpy(d->qpos, initial_q, 30*sizeof(double));
      std::memcpy(d->qvel, initial_v, 30*sizeof(double));
      mj_forward(m, d);
      frames[i] = 0;
    }
    std::memcpy(q+i*30, d->qpos, 30*sizeof(double));
    std::memcpy(v+i*30, d->qvel, 30*sizeof(double));
    mj_getState(m, d, states+i*width, spec);
  }
  return 0;
}
