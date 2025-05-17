#pragma once                // header guard

// ───────── TILE SIZES ─────────
#ifndef Tx
#define Tx 8
#endif

#ifndef Ty
#define Ty 8
#endif

#ifndef Tn
#define Tn 16
#endif

#ifndef Ti
#define Ti 16
#endif

#ifndef Tnn
#define Tnn 32          // outer-N tile
#endif

#ifndef Tii
#define Tii 32          // outer-C tile
#endif

// ───────── LAUNCH CFG ─────────
#ifndef BX
#define BX 16
#endif

#ifndef BY
#define BY 8
#endif

#ifndef BZ
#define BZ 4
#endif

// ───────── STRIDE / MISC ──────
#ifndef Sx
#define Sx 1
#endif

#ifndef Sy
#define Sy 1
#endif
