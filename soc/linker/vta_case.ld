INCLUDE generated/output_format.ld
ENTRY(_start)
INCLUDE generated/regions.ld

SECTIONS
{
	.text : {
		_ftext = .;
		KEEP(*(.isr_vector))
		*crt0*(.text)
		KEEP(*crt0*(.text))
		KEEP(*(.text.isr))
		*(.text .stub .text.* .gnu.linkonce.t.*)
		_etext = .;
	} > rom

	.rodata : {
		. = ALIGN(8); _frodata = .;
		*(.rodata .rodata.* .gnu.linkonce.r.*)
		*(.rodata1) *(.got .got.*) *(.toc .toc.*)
		FILL(0); . = ALIGN(8); _erodata = .;
	} > rom

	.rodata_ai : {
		. = ALIGN(8); _frodata_ai = .;
		KEEP(*(.rodata_ai .rodata_ai.*))
		. = ALIGN(8); _erodata_ai = .;
	} > rom

	.commands : {
		PROVIDE_HIDDEN(__bios_cmd_start = .);
		KEEP(*(.bios_cmd))
		PROVIDE_HIDDEN(__bios_cmd_end = .);
	} > rom

	.init : {
		PROVIDE_HIDDEN(__bios_init_start = .);
		KEEP(*(.bios_init))
		PROVIDE_HIDDEN(__bios_init_end = .);
	} > rom

	.data_ai : {
		. = ALIGN(8); _fdata = .; _fdata_ai = .;
		KEEP(*(.data_ai .data_ai.init .data_ai.expected))
		. = ALIGN(8); _edata_ai_init = .;
		KEEP(*(.data_ai.out))
		. = ALIGN(8); _edata_ai = .;
	} > sram AT > rom
	_fdata_ai_rom = LOADADDR(.data_ai);

	.data : {
		. = ALIGN(8);
		*(.data .data.* .gnu.linkonce.d.*)
		*(.data1) *(.sdata .sdata.* .gnu.linkonce.s.*)
		FILL(0); . = ALIGN(8); _edata = .;
	} > sram AT > rom

	.bss : {
		. = ALIGN(8); _fbss = .;
		*(.dynsbss) *(.sbss .sbss.* .gnu.linkonce.sb.*) *(.scommon)
		*(.dynbss) *(.bss .bss.* .gnu.linkonce.b.*) *(COMMON)
		. = ALIGN(8); _ebss = .; _end = .;
	} > sram

	/DISCARD/ : { *(.eh_frame) *(.comment) }
}

PROVIDE(_fstack = ORIGIN(sram) + LENGTH(sram));
PROVIDE(_fdata_rom = LOADADDR(.data_ai));
PROVIDE(_edata_rom = LOADADDR(.data) + SIZEOF(.data));
