// SPDX-License-Identifier: GPL-2.0-only
// Modified 2026-11-24 for split wave64 readback and DXIL size regression coverage.
// Adapted from akimfomin's synthetic probe, posted under GPL-2.0-only:
// https://github.com/xcl-dev/xcl/pull/1748#issuecomment-7145265103
use xcl::gpu::dxil::kernel::{translate, KernelOptions};
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::{runner, HEAP_WINDOWS};
use xcl::gpu::uma::Uma;

const BASE: usize = 0x100;
const WINDOWS: usize = 4;
const LOADS: usize = 16;

fn program(loads: usize) -> Vec<u8> {
    let mut code = Vec::new();
    for i in 0..loads {
        code.extend(encode("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[
            ("VDATA", (16 + i) as u64), ("VADDR", 0), ("SRSRC", (3 + i % WINDOWS) as u64),
            ("SOFFSET", 128), ("IDXEN", 1), ("OFFSET", (16 * i) as u64),
        ], None).unwrap());
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    code
}

fn half_mantissa(window: usize, index: usize) -> u32 {
    ((3 * index + 211 * window + 1) & 1023) as u32
}

fn check_output(dxil: &[u8], loads: usize, lanes: usize) {
    let arena = Uma::with_windows(WINDOWS, 1 << 20).unwrap();
    let mut state = runner::KernelState::default();
    for (lane, regs) in state.vgpr.iter_mut().enumerate() {
        regs[0] = lane as u32;
    }
    let dwords = loads * 4 + lanes;
    for window in 0..WINDOWS {
        let address = arena.window(window).0 + BASE;
        // 16_16_FLOAT with identity selection and a 4-byte stride.
        let desc = [address as u32, ((address >> 32) as u32 & 0xffff) | 4 << 16,
            dwords as u32, 29 << 12 | 7 << 9 | 6 << 6 | 5 << 3 | 4];
        state.sgpr[12 + 4 * window..16 + 4 * window].copy_from_slice(&desc);
        for index in 0..dwords {
            let half = 0x3c00 | half_mantissa(window, index);
            // The arena owns a committed 1 MiB region. These aligned writes stay
            // within its first 4 KiB and finish before the GPU reads the regions.
            unsafe { ((address + 4 * index) as *mut u32).write(half) };
        }
    }
    let output = runner::run_in_arena(dxil, &state, &arena).unwrap();
    for lane in 0..lanes {
        for i in 0..loads {
            let expected = 0x3f80_0000 | half_mantissa(i % WINDOWS, 4 * i + lane) << 13;
            assert_eq!(output.vgpr[lane][16 + i], expected, "lane {lane}, load {i}");
        }
    }
}

#[test]
fn repeated_format_loads_stay_compact_and_return_the_expected_values() {
    if !runner::wave32_supported().unwrap_or(false) {
        runner::skip_test("no GPU with wave32");
        return;
    }
    let code = program(LOADS);
    let mut sizes = Vec::new();
    for (wave64, lanes) in [(false, 32), (true, 64)] {
        for binding in [0, HEAP_WINDOWS] {
            let options = KernelOptions {
                wave64,
                split: wave64,
                mem_windows: WINDOWS as u32 | binding,
                sampler_base: 0,
            };
            let dxil = translate(&code, options).unwrap();
            check_output(&dxil, LOADS, lanes);
            println!("wave64={wave64} heap={} bytes={} checked={}", binding != 0, dxil.len(), LOADS * lanes);
            sizes.push(dxil.len());
        }
    }
    // On #1748 the wave32 form fell from 52–60 kB to 15–17 kB for this fixture.
    assert!(sizes.iter().all(|&bytes| bytes < 32 * 1024), "formatted-load DXIL bytes: {sizes:?}");
}
