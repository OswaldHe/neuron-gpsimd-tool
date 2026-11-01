/* Declare a named GPSIMD kernel for the trn2+/trn3 kernel-library loader (libnrtucode_extisa):
 * an ELF note, owner "Neuron", type 2, desc = { u32 entry_address; char name[]; }.
 * The entry address is written as a placeholder and patched post-link by klpatch.py. */
#pragma once
#define NEURON_KERNEL(fn)                                                   \
    __asm__(".pushsection .note.neuron,\"a\",@note\n"                     \
            ".balign 4\n"                                                   \
            ".word 7\n"              /* namesz: "Neuron\0" */               \
            ".word 2f - 1f\n"        /* descsz */                           \
            ".word 2\n"              /* type 2 = named kernel */            \
            ".asciz \"Neuron\"\n"                                           \
            ".balign 4\n"                                                   \
            "1: .word 0xdeadbeef\n"  /* entry address, patched post-link */ \
            ".asciz \"" #fn "\"\n"                                          \
            "2: .balign 4\n"                                                \
            ".popsection\n")
