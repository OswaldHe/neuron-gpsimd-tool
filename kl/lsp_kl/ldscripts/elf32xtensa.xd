/* Linker script for creating loadable libraries with 4 text and 4 data sections
   each of which can be loaded into a separate region of memory. */
ENTRY(_start)

PHDRS
{
  text0     PT_LOAD    FLAGS(0x4 | 0x1);
  notes     PT_LOAD    FLAGS(0x4);
  note      PT_NOTE    FLAGS(0x4);
  data0     PT_LOAD    FLAGS(0x4 | 0x2 | 0x1);
  scratch   PT_LOAD    FLAGS(0x2 | 0x1);
  dynamic   PT_DYNAMIC FLAGS(0x4 | 0x2);
}

SECTIONS
{
  /* Read-only sections, merged into text segment: */
  . = 0x1000000;
  .text           : ALIGN(4)
  {
    *(.got.plt* .plt*)
    KEEP (*(.init.literal))
    KEEP (*(.init))
    *(.literal .text .0.literal .0.text .stub .literal.* .text.* .gnu.linkonce.literal.* .gnu.linkonce.t.*.literal .gnu.linkonce.t.*)
    *(.clib.literal .clib.text)
    KEEP (*(.text.*personality*))
    /* .gnu.warning sections are handled specially by elf32.em.  */
    *(.gnu.warning)
    KEEP (*(.fini.literal))
    KEEP (*(.fini))
  } :text0 =0

  /* Neuron kernel-library notes: GNU build-id + one 'Neuron' type-2 note per kernel */
  .note ALIGN(4) :
  {
    KEEP (*(.note.gnu.build-id))
    KEEP (*(.note.neuron))
  } :notes :note

  /*          . = . + 0x400;     */
   . = 0x2000000;
  .gnu.version    : { *(.gnu.version) } :data0


  .gnu.version_d  : { *(.gnu.version_d) }
  .gnu.version_r  : { *(.gnu.version_r) }
  .rel.init       : { *(.rel.init) }
  .rela.init      : { *(.rela.init) }
  .rel.text       : { *(.rel.text .rel.text.* .rel.gnu.linkonce.t.*) }
  .rela.text      : { *(.rela.text .rela.text.* .rela.gnu.linkonce.t.*) }
  .rel.fini       : { *(.rel.fini) }
  .rela.fini      : { *(.rela.fini) }
  .rel.rodata     : { *(.rel.rodata .rel.rodata.* .rel.gnu.linkonce.r.*) }
  .rela.rodata    : { *(.rela.rodata .rela.rodata.* .rela.gnu.linkonce.r.*) }
  .rel.data.rel.ro   : { *(.rel.data.rel.ro*) }
  .rela.data.rel.ro   : { *(.rel.data.rel.ro*) }
  .rel.data       : { *(.rel.data .rel.data.* .rel.gnu.linkonce.d.*) }
  .rela.data      : { *(.rela.data .rela.data.* .rela.gnu.linkonce.d.*) }
  .rel.tdata	  : { *(.rel.tdata .rel.tdata.* .rel.gnu.linkonce.td.*) }
  .rela.tdata	  : { *(.rela.tdata .rela.tdata.* .rela.gnu.linkonce.td.*) }
  .rel.tbss	  : { *(.rel.tbss .rel.tbss.* .rel.gnu.linkonce.tb.*) }
  .rela.tbss	  : { *(.rela.tbss .rela.tbss.* .rela.gnu.linkonce.tb.*) }
  .rel.ctors      : { *(.rel.ctors) }
  .rela.ctors     : { *(.rela.ctors) }
  .rel.dtors      : { *(.rel.dtors) }
  .rela.dtors     : { *(.rela.dtors) }
  .rel.bss        : { *(.rel.bss .rel.bss.* .rel.gnu.linkonce.b.*) }
  .rela.bss       : { *(.rela.bss .rela.bss.* .rela.gnu.linkonce.b.*) }
  .rel.plt        : { *(.rel.plt) }
  .rela.plt       : { *(.rela.plt) }
  .clib.rodata    : { *(.clib.rodata)}
  .rtos.rodata    : { *(.rtos.rodata)}
  .clib.data      : { *(.clib.data)}
  .rtos.data      : { *(.rtos.data)}
  .clib.percpu.data   : { *(.clib.percpu.data) }
  .rtos.percpu.data   : { *(.rtos.percpu.data) }
  .rodata3        : { *(.rodata3) }
  .rodata         : { *(.rodata .rodata.* .gnu.linkonce.r.*) }
  .xt_except_table   : { KEEP (*(.xt_except_table)) }
  .eh_frame       : ONLY_IF_RO { KEEP (*(.eh_frame)) }
  .gcc_except_table   : ONLY_IF_RO { KEEP (*(.gcc_except_table)) *(.gcc_except_table.*) }
  /* Exception handling  */
  .eh_frame       : ONLY_IF_RW { KEEP (*(.eh_frame)) }
  .gcc_except_table   : ONLY_IF_RW { KEEP (*(.gcc_except_table)) *(.gcc_except_table.*) }
  /* Thread Local Storage sections  */
  .tdata	  : { *(.tdata .tdata.* .gnu.linkonce.td.*) }
  .tbss		  : { *(.tbss .tbss.* .gnu.linkonce.tb.*) *(.tcommon) }
  /* Ensure the __preinit_array_start label is properly aligned.  We
     could instead move the label definition inside the section, but
     the linker would then create the section even if it turns out to
     be empty, which isn't pretty.  */
  . = ALIGN(32 / 8);
  PROVIDE (__preinit_array_start = .);
  .preinit_array     : { KEEP (*(.preinit_array)) }
  PROVIDE (__preinit_array_end = .);
  PROVIDE (__init_array_start = .);
  .init_array     : { KEEP (*(.init_array)) }
  PROVIDE (__init_array_end = .);
  PROVIDE (__fini_array_start = .);
  .fini_array     : { KEEP (*(.fini_array)) }
  PROVIDE (__fini_array_end = .);
  .ctors          :
  {
    /* gcc uses crtbegin.o to find the start of
       the constructors, so we make sure it is
       first.  Because this is a wildcard, it
       doesn't matter if the user does not
       actually link against crtbegin.o; the
       linker won't look for a file to match a
       wildcard.  The wildcard also means that it
       doesn't matter which directory crtbegin.o
       is in.  */
    KEEP (*crtbegin*.o(.ctors))
    /* We don't want to include the .ctor section from
       from the crtend.o file until after the sorted ctors.
       The .ctor section from the crtend file contains the
       end of ctors marker and it must be last */
    KEEP (*(EXCLUDE_FILE (*crtend*.o ) .ctors))
    KEEP (*(SORT(.ctors.*)))
    KEEP (*(.ctors))
  }
  .dtors          :
  {
    KEEP (*crtbegin*.o(.dtors))
    KEEP (*(EXCLUDE_FILE (*crtend*.o ) .dtors))
    KEEP (*(SORT(.dtors.*)))
    KEEP (*(.dtors))
  }
  .jcr            : { KEEP (*(.jcr)) }
  .data.rel.ro : { *(.data.rel.ro.local) *(.data.rel.ro*) }
   
    /* PROVIDE (__0_data_start = .); */
        .data           :
  {
 __data_start = .;
    *(.data .data.* .0.data .gnu.linkonce.d.*)
    KEEP (*(.gnu.linkonce.d.*personality*))
    SORT(CONSTRUCTORS)
  }
  __bss_start = .;
  .bss            :
  {
   *(.dynbss)
   *(.bss .bss.* .gnu.linkonce.b.*)
   *(.clib.bss)
   *(.clib.percpu.bss)
   *(.rtos.bss)
   *(.rtos.percpu.bss)
   *(COMMON)
   /* Align here to ensure that the .bss section occupies space up to
      _end.  Align after .bss to ensure correct alignment even if the
      .bss section disappears because there are no input sections.  */
   . = ALIGN(32 / 8);
  }
  . = ALIGN(32 / 8);
  _end = .;
  PROVIDE (end = .);
  /* Stabs debugging sections.  */
  .stab          0 : { *(.stab) } : NONE
  .stabstr       0 : { *(.stabstr) }
  .stab.excl     0 : { *(.stab.excl) }
  .stab.exclstr  0 : { *(.stab.exclstr) }
  .stab.index    0 : { *(.stab.index) }
  .stab.indexstr 0 : { *(.stab.indexstr) }
  .comment       0 : { *(.comment) }
  /* DWARF debug sections.
     Symbols in the DWARF debugging sections are relative to the beginning
     of the section so we begin them at 0.  */
  /* DWARF 3 */
  .debug          0 : { *(.debug) }
  .line           0 : { *(.line) }
  /* GNU DWARF 3 extensions */
  .debug_srcinfo  0 : { *(.debug_srcinfo) }
  .debug_sfnames  0 : { *(.debug_sfnames) }
  /* DWARF 3.3 and DWARF 2 */
  .debug_aranges  0 : { *(.debug_aranges) }
  .debug_pubnames 0 : { *(.debug_pubnames) }
  /* DWARF 2 */
  .debug_info     0 : { *(.debug_info .gnu.linkonce.wi.*) }
  .debug_abbrev   0 : { *(.debug_abbrev) }
  .debug_line     0 : { *(.debug_line) }
  .debug_frame    0 : { *(.debug_frame) }
  .debug_str      0 : { *(.debug_str) }
  .debug_loc      0 : { *(.debug_loc) }
  .debug_macinfo  0 : { *(.debug_macinfo) }
  /* SGI/MIPS DWARF 2 extensions */
  .debug_weaknames 0 : { *(.debug_weaknames) }
  .debug_funcnames 0 : { *(.debug_funcnames) }
  .debug_typenames 0 : { *(.debug_typenames) }
  .debug_varnames  0 : { *(.debug_varnames) }
  .xt.lit         0 : { *(.xt.lit .xt.lit.* .gnu.linkonce.p.*) }
  .xt.insn        0 : { *(.xt.insn .gnu.linkonce.x.*) }
  .xt.prop        0 : { *(.xt.prop .gnu.linkonce.prop.*) }

. = 0x3000000;

  /DISCARD/ : { *(.note.GNU-stack) *(.interp) }
  .dynamic : 
  {
    *(.dynamic)
    LONG(0x70000002)
    *(.dyninfo)
    KEEP(*(.dyninfo))
  } :dynamic :scratch
  .hash           : { *(.hash) } :scratch
  .dynsym         : { *(.dynsym) } 
  .dynstr         : { *(.dynstr) } 

  .rel.got        : { *(.rel.got) } 
  .rela.got       : { *(.rela.got) } 
  .got            : { *(.got) } 
  .got.loc        : { *(.got.loc) }


.scratch.data :
 {
   __scratch.data_start = .;
    *(.scratch.data)
   __scratch.data_end = .;
 } :scratch

}

