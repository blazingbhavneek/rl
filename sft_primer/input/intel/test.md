### [Objective-C Garbage Collection Module Flags Metadata](#id2165)[¶](#objective-c-garbage-collection-module-flags-metadata "Link to this heading")

On the Mach-O platform, Objective-C stores metadata about garbage collection in a special section called “image info”. The metadata consists of a version number and a bitmask specifying what types of garbage collection are supported (if any) by the file. If two or more modules are linked together their garbage collection metadata needs to be merged rather than appended together.

The Objective-C garbage collection module flags metadata consists of the following key-value pairs:

 

Key

Value

`Objective-C Version`

**\[Required\]** — The Objective-C ABI version. Valid values are 1 and 2.

`Objective-C Image Info Version`

**\[Required\]** — The version of the image info section. Currently always 0.

`Objective-C Image Info Section`

**\[Required\]** — The section to place the metadata. Valid values are `"__OBJC, __image_info, regular"` for Objective-C ABI version 1, and `"__DATA,__objc_imageinfo, regular, no_dead_strip"` for Objective-C ABI version 2.

`Objective-C Garbage Collection`

**\[Required\]** — Specifies whether garbage collection is supported or not. Valid values are 0, for no garbage collection, and 2, for garbage collection supported.

`Objective-C GC Only`

**\[Optional\]** — Specifies that only garbage collection is supported. If present, its value must be 6. This flag requires that the `Objective-C Garbage Collection` flag have the value 2.

Some important flag interactions:

*   If a module with `Objective-C Garbage Collection` set to 0 is merged with a module with `Objective-C Garbage Collection` set to 2, then the resulting module has the `Objective-C Garbage Collection` flag set to 0.
    
*   A module with `Objective-C Garbage Collection` set to 0 cannot be merged with a module with `Objective-C GC Only` set to 6.
    

### [C type width Module Flags Metadata](#id2166)[¶](#c-type-width-module-flags-metadata "Link to this heading")

The ARM backend emits a section into each generated object file describing the options that it was compiled with (in a compiler-independent way) to prevent linking incompatible objects, and to allow automatic library selection. Some of these options are not visible at the IR level, namely wchar\_t width and enum width.

To pass this information to the backend, these options are encoded in module flags metadata, using the following key-value pairs:

 

Key

Value

short\_wchar

*   0 — sizeof(wchar\_t) == 4
    
*   1 — sizeof(wchar\_t) == 2
    

short\_enum

*   0 — Enums are at least as large as an `int`.
    
*   1 — Enums are stored in the smallest integer type which can represent all of its values.
    

For example, the following metadata section specifies that the module was compiled with a `wchar_t` width of 4 bytes, and the underlying type of an enum is the smallest type which can represent all of its values:

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"short\_wchar", i32 1}
!1 = !{i32 1, !"short\_enum", i32 0}

### [Stack Alignment Metadata](#id2167)[¶](#stack-alignment-metadata "Link to this heading")

Changes the default stack alignment from the target ABI’s implicit default stack alignment. Takes an i32 value in bytes. It is considered an error to link two modules together with different values for this metadata.

For example:

> !llvm.module.flags = !{!0} !0 = !{i32 1, !”override-stack-alignment”, i32 8}

This will change the stack alignment to 8B.

[Embedded Objects Names Metadata](#id2168)[¶](#embedded-objects-names-metadata "Link to this heading")
------------------------------------------------------------------------------------------------------

Offloading compilations need to embed device code into the host section table to create a fat binary. This metadata node references each global that will be embedded in the module. The primary use for this is to make referencing these globals more efficient in the IR. The metadata references nodes containing pointers to the global to be embedded followed by the section name it will be stored at:

!llvm.embedded.objects = !{!0}
!0 = !{ptr @object, !".section"}

[Automatic Linker Flags Named Metadata](#id2169)[¶](#automatic-linker-flags-named-metadata "Link to this heading")
------------------------------------------------------------------------------------------------------------------

Some targets support embedding of flags to the linker inside individual object files. Typically this is used in conjunction with language extensions which allow source files to contain linker command-line options, and have these automatically be transmitted to the linker via object files.

These flags are encoded in the IR using named metadata with the name `!llvm.linker.options`. Each operand is expected to be a metadata node which should be a list of other metadata nodes, each of which should be a list of metadata strings defining linker options.

For example, the following metadata section specifies two separate sets of linker options, presumably to link against `libz` and the `Cocoa` framework:

!0 = !{ !"-lz" }
!1 = !{ !"-framework", !"Cocoa" }
!llvm.linker.options = !{ !0, !1 }

The metadata encoding as lists of lists of options, as opposed to a collapsed list of options, is chosen so that the IR encoding can use multiple option strings to specify e.g., a single library, while still having that specifier be preserved as an atomic element that can be recognized by a target-specific assembly writer or object file emitter.

Each individual option is required to be either a valid option for the target’s linker, or an option that is reserved by the target-specific assembly writer or object file emitter. No other aspect of these options is defined by the IR.

[Dependent Libs Named Metadata](#id2170)[¶](#dependent-libs-named-metadata "Link to this heading")
--------------------------------------------------------------------------------------------------

Some targets support embedding of strings into object files to indicate a set of libraries to add to the link. Typically this is used in conjunction with language extensions which allow source files to explicitly declare the libraries they depend on, and have these automatically be transmitted to the linker via object files.

The list is encoded in the IR using named metadata with the name `!llvm.dependent-libraries`. Each operand is expected to be a metadata node which should contain a single string operand.

For example, the following metadata section contains two library specifiers:

!0 = !{!"a library specifier"}
!1 = !{!"another library specifier"}
!llvm.dependent-libraries = !{ !0, !1 }

Each library specifier will be handled independently by the consuming linker. The effect of the library specifiers are defined by the consuming linker.

[‘`llvm.errno.tbaa`’ Named Metadata](#id2171)[¶](#llvm-errno-tbaa-named-metadata "Link to this heading")
--------------------------------------------------------------------------------------------------------

The module-level `!llvm.errno.tbaa` metadata specifies the TBAA nodes used for accessing `errno`. These nodes are guaranteed to represent int-compatible accesses according to C/C++ strict aliasing rules. This should let LLVM alias analyses to reason about aliasing with `errno` when calling library functions that may set `errno`, allowing optimizations such as store-to-load forwarding across such routines.

For example, the following is a valid metadata specifying the TBAA information for an integer access:

!llvm.errno.tbaa = !{!0}
!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}

Multiple TBAA operands are allowed to support merging of modules that may use different TBAA hierarchies (e.g., when mixing C and C++).

[ThinLTO Summary](#id2172)[¶](#thinlto-summary "Link to this heading")
----------------------------------------------------------------------

Compiling with [ThinLTO](https://clang.llvm.org/docs/ThinLTO.html) causes the building of a compact summary of the module that is emitted into the bitcode. The summary is emitted into the LLVM assembly and identified in syntax by a caret (’`^`’).

The summary is parsed into a bitcode output, along with the Module IR, via the “`llvm-as`” tool. Tools that parse the Module IR for the purposes of optimization (e.g., “`clang -x ir`” and “`opt`”), will ignore the summary entries (just as they currently ignore summary entries in a bitcode input file).

Eventually, the summary will be parsed into a ModuleSummaryIndex object under the same conditions where summary index is currently built from bitcode. Specifically, tools that test the Thin Link portion of a ThinLTO compile (i.e., llvm-lto and llvm-lto2), or when parsing a combined index for a distributed ThinLTO backend via clang’s “`-fthinlto-index=<>`” flag (this part is not yet implemented, use llvm-as to create a bitcode object before feeding into thin link tools for now).

There are currently 3 types of summary entries in the LLVM assembly: [module paths](#module-path-summary), [global values](#gv-summary), and [type identifiers](#typeid-summary).

### [Module Path Summary Entry](#id2173)[¶](#module-path-summary-entry "Link to this heading")

Each module path summary entry lists a module containing global values included in the summary. For a single IR module there will be one such entry, but in a combined summary index produced during the thin link, there will be one module path entry per linked module with summary.

Example:

^0 = module: (path: "/path/to/file.o", hash: (2468601609, 1329373163, 1565878005, 638838075, 3148790418))

The `path` field is a string path to the bitcode file, and the `hash` field is the 160-bit SHA-1 hash of the IR bitcode contents, used for incremental builds and caching.

### [Global Value Summary Entry](#id2174)[¶](#global-value-summary-entry "Link to this heading")

Each global value summary entry corresponds to a global value defined or referenced by a summarized module.

Example:

^4 = gv: (name: "f"\[, summaries: (Summary)\[, (Summary)\]\*\]?) ; guid = 14740650423002898831

For declarations, there will not be a summary list. For definitions, a global value will contain a list of summaries, one per module containing a definition. There can be multiple entries in a combined summary index for symbols with weak linkage.

Each `Summary` format will depend on whether the global value is a [function](#function-summary), [variable](#variable-summary), or [alias](#alias-summary).

#### [Function Summary](#id2175)[¶](#function-summary "Link to this heading")

If the global value is a function, the `Summary` entry will look like:

function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 2\[, FuncFlags\]?\[, Calls\]?\[, TypeIdInfo\]?\[, Params\]?\[, Refs\]?

The `module` field includes the summary entry id for the module containing this definition, and the `flags` field contains information such as the linkage type, a flag indicating whether it is legal to import the definition, whether it is globally live and whether the linker resolved it to a local definition (the latter two are populated during the thin link). The `insts` field contains the number of IR instructions in the function. Finally, there are several optional fields: [FuncFlags](#funcflags-summary), [Calls](#calls-summary), [TypeIdInfo](#typeidinfo-summary), [Params](#params-summary), [Refs](#refs-summary).

#### [Global Variable Summary](#id2176)[¶](#global-variable-summary "Link to this heading")

If the global value is a variable, the `Summary` entry will look like:

variable: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0)\[, Refs\]?

The variable entry contains a subset of the fields in a [function summary](#function-summary), see the descriptions there.

#### [Alias Summary](#id2177)[¶](#alias-summary "Link to this heading")

If the global value is an alias, the `Summary` entry will look like:

alias: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), aliasee: ^2)

The `module` and `flags` fields are as described for a [function summary](#function-summary). The `aliasee` field contains a reference to the global value summary entry of the aliasee.

#### [Function Flags](#id2178)[¶](#function-flags "Link to this heading")

The optional `FuncFlags` field looks like:

funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 0, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0)

If unspecified, flags are assumed to hold the conservative `false` value of `0`.

#### [Calls](#id2179)[¶](#calls "Link to this heading")

The optional `Calls` field looks like:

calls: ((Callee)\[, (Callee)\]\*)

where each `Callee` looks like:

callee: ^1\[, hotness: None\]?\[, relbf: 0\]?

The `callee` refers to the summary entry id of the callee. At most one of `hotness` (which can take the values `Unknown`, `Cold`, `None`, `Hot`, and `Critical`), and `relbf` (which holds the integer branch frequency relative to the entry frequency, scaled down by 2^8) may be specified. The defaults are `Unknown` and `0`, respectively.

#### [Params](#id2180)[¶](#params "Link to this heading")

The optional `Params` is used by `StackSafety` and looks like:

Params: ((Param)\[, (Param)\]\*)

where each `Param` describes pointer parameter access inside of the function and looks like:

param: 4, offset: \[0, 5\]\[, calls: ((Callee)\[, (Callee)\]\*)\]?

where the first `param` is the number of the parameter it describes, `offset` is the inclusive range of offsets from the pointer parameter to bytes which can be accessed by the function. This range does not include accesses by function calls from `calls` list.

where each `Callee` describes how parameter is forwarded into other functions and looks like:

callee: ^3, param: 5, offset: \[-3, 3\]

The `callee` refers to the summary entry id of the callee, `param` is the number of the callee parameter which points into the callers parameter with offset known to be inside of the `offset` range. `calls` will be consumed and removed by thin link stage to update `Param::offset` so it covers all accesses possible by `calls`.

Pointer parameter without corresponding `Param` is considered unsafe and we assume that access with any offset is possible.

Example:

If we have the following function:

define i64 @foo(ptr %0, ptr %1, ptr %2, i8 %3) {
  store ptr %1, ptr @x
  %5 = getelementptr inbounds i8, ptr %2, i64 5
  %6 = load i8, ptr %5
  %7 = getelementptr inbounds i8, ptr %2, i8 %3
  tail call void @bar(i8 %3, ptr %7)
  %8 = load i64, ptr %0
  ret i64 %8
}

We can expect the record like this:

params: ((param: 0, offset: \[0, 7\]),(param: 2, offset: \[5, 5\], calls: ((callee: ^3, param: 1, offset: \[-128, 127\]))))

The function may access just 8 bytes of the parameter %0 . `calls` is empty, so the parameter is either not used for function calls or `offset` already covers all accesses from nested function calls. Parameter %1 escapes, so access is unknown. The function itself can access just a single byte of the parameter %2. Additional access is possible inside of the `@bar` or `^3`. The function adds signed offset to the pointer and passes the result as the argument %1 into `^3`. This record itself does not tell us how `^3` will access the parameter. Parameter %3 is not a pointer.

#### [Refs](#id2181)[¶](#refs "Link to this heading")

The optional `Refs` field looks like:

refs: ((Ref)\[, (Ref)\]\*)

where each `Ref` contains a reference to the summary id of the referenced value (e.g., `^1`).

#### [TypeIdInfo](#id2182)[¶](#typeidinfo "Link to this heading")

The optional `TypeIdInfo` field, used for [Control Flow Integrity](https://clang.llvm.org/docs/ControlFlowIntegrity.html), looks like:

typeIdInfo: \[(TypeTests)\]?\[, (TypeTestAssumeVCalls)\]?\[, (TypeCheckedLoadVCalls)\]?\[, (TypeTestAssumeConstVCalls)\]?\[, (TypeCheckedLoadConstVCalls)\]?

These optional fields have the following forms:

##### TypeTests[¶](#typetests "Link to this heading")

typeTests: (TypeIdRef\[, TypeIdRef\]\*)

Where each `TypeIdRef` refers to a [type id](#typeid-summary) by summary id or `GUID`.

##### TypeTestAssumeVCalls[¶](#typetestassumevcalls "Link to this heading")

typeTestAssumeVCalls: (VFuncId\[, VFuncId\]\*)

Where each VFuncId has the format:

vFuncId: (TypeIdRef, offset: 16)

Where each `TypeIdRef` refers to a [type id](#typeid-summary) by summary id or `GUID` preceded by a `guid:` tag.

##### TypeCheckedLoadVCalls[¶](#typecheckedloadvcalls "Link to this heading")

typeCheckedLoadVCalls: (VFuncId\[, VFuncId\]\*)

Where each VFuncId has the format described for `TypeTestAssumeVCalls`.

##### TypeTestAssumeConstVCalls[¶](#typetestassumeconstvcalls "Link to this heading")

typeTestAssumeConstVCalls: (ConstVCall\[, ConstVCall\]\*)

Where each ConstVCall has the format:

(VFuncId, args: (Arg\[, Arg\]\*))

and where each VFuncId has the format described for `TypeTestAssumeVCalls`, and each Arg is an integer argument number.

##### TypeCheckedLoadConstVCalls[¶](#typecheckedloadconstvcalls "Link to this heading")

typeCheckedLoadConstVCalls: (ConstVCall\[, ConstVCall\]\*)

Where each ConstVCall has the format described for `TypeTestAssumeConstVCalls`.

### [Type ID Summary Entry](#id2183)[¶](#type-id-summary-entry "Link to this heading")

Each type id summary entry corresponds to a type identifier resolution which is generated during the LTO link portion of the compile when building with [Control Flow Integrity](https://clang.llvm.org/docs/ControlFlowIntegrity.html), so these are only present in a combined summary index.

Example:

^4 = typeid: (name: "\_ZTS1A", summary: (typeTestRes: (kind: allOnes, sizeM1BitWidth: 7\[, alignLog2: 0\]?\[, sizeM1: 0\]?\[, bitMask: 0\]?\[, inlineBits: 0\]?)\[, WpdResolutions\]?)) ; guid = 7004155349499253778

The `typeTestRes` gives the type test resolution `kind` (which may be `unsat`, `byteArray`, `inline`, `single`, or `allOnes`), and the `size-1` bit width. It is followed by optional flags, which default to 0, and an optional WpdResolutions (whole program devirtualization resolution) field that looks like:

wpdResolutions: ((offset: 0, WpdRes)\[, (offset: 1, WpdRes)\]\*

where each entry is a mapping from the given byte offset to the whole-program devirtualization resolution WpdRes, that has one of the following formats:

wpdRes: (kind: branchFunnel)
wpdRes: (kind: singleImpl, singleImplName: "\_ZN1A1nEi")
wpdRes: (kind: indir)

Additionally, each wpdRes has an optional `resByArg` field, which describes the resolutions for calls with all constant integer arguments:

resByArg: (ResByArg\[, ResByArg\]\*)

where ResByArg is:

args: (Arg\[, Arg\]\*), byArg: (kind: UniformRetVal\[, info: 0\]\[, byte: 0\]\[, bit: 0\])

Where the `kind` can be `Indir`, `UniformRetVal`, `UniqueRetVal` or `VirtualConstProp`. The `info` field is only used if the kind is `UniformRetVal` (indicates the uniform return value), or `UniqueRetVal` (holds the return value associated with the unique vtable (0 or 1)). The `byte` and `bit` fields are only used if the target does not support the use of absolute symbols to store constants.

[Intrinsic Global Variables](#id2184)[¶](#intrinsic-global-variables "Link to this heading")
--------------------------------------------------------------------------------------------

LLVM has a number of “magic” global variables that contain data that affect code generation or other IR semantics. These are documented here. All globals of this sort should have a section specified as “`llvm.metadata`”. This section and all globals that start with “`llvm.`” are reserved for use by LLVM.

### [The ‘`llvm.used`’ Global Variable](#id2185)[¶](#the-llvm-used-global-variable "Link to this heading")

The `@llvm.used` global is an array which has [appending linkage](#linkage-appending). This array contains a list of pointers to named global variables, functions and aliases which may optionally have a pointer cast formed of bitcast or getelementptr. For example, a legal use of it is:

@X \= global i8 4
@Y \= global i32 123

@llvm.used \= appending global \[2 x ptr\] \[
   ptr @X,
   ptr @Y
\], section "llvm.metadata"

If a symbol appears in the `@llvm.used` list, then the compiler, assembler, and linker are required to treat the symbol as if there is a reference to the symbol that it cannot see (which is why they have to be named). For example, if a variable has internal linkage and no references other than that from the `@llvm.used` list, it cannot be deleted. This is commonly used to represent references from inline asms and other things the compiler cannot “see”, and corresponds to “`attribute((used))`” in GNU C.

On some targets, the code generator must emit a directive to the assembler or object file to prevent the assembler and linker from removing the symbol.

### [The ‘`llvm.compiler.used`’ Global Variable](#id2186)[¶](#the-llvm-compiler-used-global-variable "Link to this heading")

The `@llvm.compiler.used` directive is the same as the `@llvm.used` directive, except that it only prevents the compiler from touching the symbol. On targets that support it, this allows an intelligent linker to optimize references to the symbol without being impeded as it would be by `@llvm.used`.

This is a rare construct that should only be used in rare circumstances, and should not be exposed to source languages.

### [The ‘`llvm.global_ctors`’ Global Variable](#id2187)[¶](#the-llvm-global-ctors-global-variable "Link to this heading")

%0 \= type { i32, ptr, ptr }
@llvm.global\_ctors \= appending global \[1 x %0\] \[%0 { i32 65535, ptr @ctor, ptr @data }\]

The `@llvm.global_ctors` array contains a list of constructor functions, priorities, and an associated global or function. The functions referenced by this array will be called in ascending order of priority (i.e., lowest first) when the module is loaded. The order of functions with the same priority is not defined.

If the third field is non-null, and points to a global variable or function, the initializer function will only run if the associated data from the current module is not discarded. On ELF the referenced global variable or function must be in a comdat.

### [The ‘`llvm.global_dtors`’ Global Variable](#id2188)[¶](#the-llvm-global-dtors-global-variable "Link to this heading")

%0 \= type { i32, ptr, ptr }
@llvm.global\_dtors \= appending global \[1 x %0\] \[%0 { i32 65535, ptr @dtor, ptr @data }\]

The `@llvm.global_dtors` array contains a list of destructor functions, priorities, and an associated global or function. The functions referenced by this array will be called in descending order of priority (i.e., highest first) when the module is unloaded. The order of functions with the same priority is not defined.

If the third field is non-null, and points to a global variable or function, the destructor function will only run if the associated data from the current module is not discarded. On ELF the referenced global variable or function must be in a comdat.

[Instruction Reference](#id2189)[¶](#instruction-reference "Link to this heading")
----------------------------------------------------------------------------------

The LLVM instruction set consists of several different classifications of instructions: [terminator instructions](#terminators), [binary instructions](#binaryops), [bitwise binary instructions](#bitwiseops), [memory instructions](#memoryops), and [other instructions](#otherops). There are also [debug records](#debugrecords), which are not instructions themselves but are printed interleaved with instructions to describe changes in the state of the program’s debug information at each position in the program’s execution.

### [Terminator Instructions](#id2190)[¶](#terminator-instructions "Link to this heading")

As mentioned [previously](#functionstructure), every basic block in a program ends with a “Terminator” instruction, which indicates which block should be executed after the current block is finished. These terminator instructions typically yield a ‘`void`’ value: they produce control flow, not values (the one exception being the ‘[invoke](#i-invoke)’ instruction).

The terminator instructions are: ‘[ret](#i-ret)’, ‘[br](#i-br)’, ‘[switch](#i-switch)’, ‘[indirectbr](#i-indirectbr)’, ‘[invoke](#i-invoke)’, ‘[callbr](#i-callbr)’ ‘[resume](#i-resume)’, ‘[catchswitch](#i-catchswitch)’, ‘[catchret](#i-catchret)’, ‘[cleanupret](#i-cleanupret)’, and ‘[unreachable](#i-unreachable)’.

#### [‘`ret`’ Instruction](#id2191)[¶](#ret-instruction "Link to this heading")

##### Syntax:[¶](#id34 "Link to this heading")

ret <type\> <value\>       ; Return a value from a non\-void function
ret void                 ; Return from void function

##### Overview:[¶](#overview "Link to this heading")

The ‘`ret`’ instruction is used to return control flow (and optionally a value) from a function back to the caller.

There are two forms of the ‘`ret`’ instruction: one that returns a value and then causes control flow, and one that just causes control flow to occur.

##### Arguments:[¶](#arguments "Link to this heading")

The ‘`ret`’ instruction optionally accepts a single argument, the return value. The type of the return value must be a ‘[first class](#t-firstclass)’ type.

A function is not [well formed](#wellformed) if it has a non-void return type and contains a ‘`ret`’ instruction with no return value or a return value with a type that does not match its type, or if it has a void return type and contains a ‘`ret`’ instruction with a return value.

##### Semantics:[¶](#id35 "Link to this heading")

When the ‘`ret`’ instruction is executed, control flow returns back to the calling function’s context. If the caller is a “[call](#i-call)” instruction, execution continues at the instruction after the call. If the caller was an “[invoke](#i-invoke)” instruction, execution continues at the beginning of the “normal” destination block. If the instruction returns a value, that value shall set the call or invoke instruction’s return value.

##### Example:[¶](#example "Link to this heading")

ret i32 5                       ; Return an integer value of 5
ret void                        ; Return from a void function
ret { i32, i8 } { i32 4, i8 2 } ; Return a struct of values 4 and 2

#### [‘`br`’ Instruction](#id2192)[¶](#br-instruction "Link to this heading")

##### Syntax:[¶](#id36 "Link to this heading")

br i1 <cond\>, label <iftrue\>, label <iffalse\>
br label <dest\>          ; Unconditional branch

##### Overview:[¶](#id37 "Link to this heading")

The ‘`br`’ instruction is used to cause control flow to transfer to a different basic block in the current function. There are two forms of this instruction, corresponding to a conditional branch and an unconditional branch.

##### Arguments:[¶](#id38 "Link to this heading")

The conditional branch form of the ‘`br`’ instruction takes a single ‘`i1`’ value and two ‘`label`’ values. The unconditional form of the ‘`br`’ instruction takes a single ‘`label`’ value as a target.

##### Semantics:[¶](#id39 "Link to this heading")

Upon execution of a conditional ‘`br`’ instruction, the ‘`i1`’ argument is evaluated. If the value is `true`, control flows to the ‘`iftrue`’ `label` argument. If “cond” is `false`, control flows to the ‘`iffalse`’ `label` argument. If ‘`cond`’ is `poison` or `undef`, this instruction has undefined behavior.

##### Example:[¶](#id40 "Link to this heading")

Test:
  %cond \= icmp eq i32 %a, %b
  br i1 %cond, label %IfEqual, label %IfUnequal
IfEqual:
  ret i32 1
IfUnequal:
  ret i32 0

#### [‘`switch`’ Instruction](#id2193)[¶](#switch-instruction "Link to this heading")

##### Syntax:[¶](#id41 "Link to this heading")

switch <intty\> <value\>, label <defaultdest\> \[ <intty\> <val\>, label <dest\> ... \]

##### Overview:[¶](#id42 "Link to this heading")

The ‘`switch`’ instruction is used to transfer control flow to one of several different places. It is a generalization of the ‘`br`’ instruction, allowing a branch to occur to one of many possible destinations.

##### Arguments:[¶](#id43 "Link to this heading")

The ‘`switch`’ instruction uses three parameters: an integer comparison value ‘`value`’, a default ‘`label`’ destination, and an array of pairs of comparison value constants and ‘`label`’s. The table is not allowed to contain duplicate constant entries.

##### Semantics:[¶](#id44 "Link to this heading")

The `switch` instruction specifies a table of values and destinations. When the ‘`switch`’ instruction is executed, this table is searched for the given value. If the value is found, control flow is transferred to the corresponding destination; otherwise, control flow is transferred to the default destination. If ‘`value`’ is `poison` or `undef`, this instruction has undefined behavior.

##### Implementation:[¶](#implementation "Link to this heading")

Depending on properties of the target machine and the particular `switch` instruction, this instruction may be code generated in different ways. For example, it could be generated as a series of chained conditional branches or with a lookup table.

##### Example:[¶](#id45 "Link to this heading")

; Emulate a conditional br instruction
%Val \= zext i1 %value to i32
switch i32 %Val, label %truedest \[ i32 0, label %falsedest \]

; Emulate an unconditional br instruction
switch i32 0, label %dest \[ \]

; Implement a jump table:
switch i32 %val, label %otherwise \[ i32 0, label %onzero
                                    i32 1, label %onone
                                    i32 2, label %ontwo \]

#### [‘`indirectbr`’ Instruction](#id2194)[¶](#indirectbr-instruction "Link to this heading")

##### Syntax:[¶](#id46 "Link to this heading")

indirectbr ptr <address\>, \[ label <dest1\>, label <dest2\>, ... \]

##### Overview:[¶](#id47 "Link to this heading")

The ‘`indirectbr`’ instruction implements an indirect branch to a label within the current function, whose address is specified by “`address`”. Address must be derived from a [blockaddress](#blockaddress) constant.

##### Arguments:[¶](#id48 "Link to this heading")

The ‘`address`’ argument is the address of the label to jump to. The rest of the arguments indicate the full set of possible destinations that the address may point to. Blocks are allowed to occur multiple times in the destination list, though this isn’t particularly useful.

This destination list is required so that dataflow analysis has an accurate understanding of the CFG.

##### Semantics:[¶](#id49 "Link to this heading")

Control transfers to the block specified in the address argument. All possible destination blocks must be listed in the label list, otherwise this instruction has undefined behavior. This implies that jumps to labels defined in other functions have undefined behavior as well. If ‘`address`’ is `poison` or `undef`, this instruction has undefined behavior.

##### Implementation:[¶](#id50 "Link to this heading")

This is typically implemented with a jump through a register.

##### Example:[¶](#id51 "Link to this heading")

indirectbr ptr %Addr, \[ label %bb1, label %bb2, label %bb3 \]

#### [‘`invoke`’ Instruction](#id2195)[¶](#invoke-instruction "Link to this heading")

##### Syntax:[¶](#id52 "Link to this heading")

<result\> \= invoke \[cconv\] \[ret attrs\] \[addrspace(<num\>)\] <ty\>|<fnty\> <fnptrval\>(<function args\>) \[fn attrs\]
              \[operand bundles\] to label <normal label\> unwind label <exception label\>

##### Overview:[¶](#id53 "Link to this heading")

The ‘`invoke`’ instruction causes control to transfer to a specified function, with the possibility of control flow transfer to either the ‘`normal`’ label or the ‘`exception`’ label. If the callee function returns with the “`ret`” instruction, control flow will return to the “normal” label. If the callee (or any indirect callees) returns via the “[resume](#i-resume)” instruction or other exception handling mechanism, control is interrupted and continued at the dynamically nearest “exception” label.

The ‘`exception`’ label is a [landing pad](https://llvm.org/docs/ExceptionHandling.html#overview) for the exception. As such, ‘`exception`’ label is required to have the “[landingpad](#i-landingpad)” instruction, which contains the information about the behavior of the program after unwinding happens, as its first non-PHI instruction. The restrictions on the “`landingpad`” instruction’s tightly couples it to the “`invoke`” instruction, so that the important information contained within the “`landingpad`” instruction can’t be lost through normal code motion.

##### Arguments:[¶](#id54 "Link to this heading")

This instruction requires several arguments:

1.  The optional “cconv” marker indicates which [calling convention](#callingconv) the call should use. If none is specified, the call defaults to using C calling conventions.
    
2.  The optional [Parameter Attributes](#paramattrs) list for return values. Only ‘`zeroext`’, ‘`signext`’, ‘`noext`’, and ‘`inreg`’ attributes are valid here.
    
3.  The optional addrspace attribute can be used to indicate the address space of the called function. If it is not specified, the program address space from the [datalayout string](#langref-datalayout) will be used.
    
4.  ‘`ty`’: the type of the call instruction itself which is also the type of the return value. Functions that return no value are marked `void`. The signature is computed based on the return type and argument types.
    
5.  ‘`fnty`’: shall be the signature of the function being invoked. The argument types must match the types implied by this signature. This is only required if the signature specifies a varargs type.
    
6.  ‘`fnptrval`’: An LLVM value containing a pointer to a function to be invoked. In most cases, this is a direct function invocation, but indirect `invoke`’s are just as possible, calling an arbitrary pointer to function value.
    
7.  ‘`function args`’: argument list whose types match the function signature argument types and parameter attributes. All arguments must be of [first class](#t-firstclass) type. If the function signature indicates the function accepts a variable number of arguments, the extra arguments can be specified.
    
8.  ‘`normal label`’: the label reached when the called function executes a ‘`ret`’ instruction.
    
9.  ‘`exception label`’: the label reached when a callee returns via the [resume](#i-resume) instruction or other exception handling mechanism.
    
10.  The optional [function attributes](#fnattrs) list.
    
11.  The optional [operand bundles](#opbundles) list.
    

##### Semantics:[¶](#id55 "Link to this heading")

This instruction is designed to operate as a standard ‘`call`’ instruction in most regards. The primary difference is that it establishes an association with a label, which is used by the runtime library to unwind the stack.

This instruction is used in languages with destructors to ensure that proper cleanup is performed in the case of either a `longjmp` or a thrown exception. Additionally, this is important for implementation of ‘`catch`’ clauses in high-level languages that support them.

For the purposes of the SSA form, the definition of the value returned by the ‘`invoke`’ instruction is deemed to occur on the edge from the current block to the “normal” label. If the callee unwinds then no return value is available.

##### Example:[¶](#id56 "Link to this heading")

%retval \= invoke i32 @Test(i32 15) to label %Continue
            unwind label %TestCleanup              ; i32:retval set
%retval \= invoke coldcc i32 %Testfnptr(i32 15) to label %Continue
            unwind label %TestCleanup              ; i32:retval set

#### [‘`callbr`’ Instruction](#id2196)[¶](#callbr-instruction "Link to this heading")

##### Syntax:[¶](#id57 "Link to this heading")

<result\> \= callbr \[cconv\] \[ret attrs\] \[addrspace(<num\>)\] <ty\>|<fnty\> <fnptrval\>(<function args\>) \[fn attrs\]
              \[operand bundles\] to label <fallthrough label\> \[indirect labels\]

##### Overview:[¶](#id58 "Link to this heading")

The ‘`callbr`’ instruction causes control to transfer to a specified function, with the possibility of control flow transfer to either the ‘`fallthrough`’ label or one of the ‘`indirect`’ labels.

This instruction can currently only be used

1.  to implement the “goto” feature of gcc style inline assembly or
    
2.  to call selected intrinsics.
    

Any other usage is an error in the IR verifier.

Note that in order to support outputs along indirect edges, LLVM may need to split critical edges, which may require synthesizing a replacement block for the `indirect labels`. Therefore, the address of a label as seen by another `callbr` instruction, or for a [blockaddress](#blockaddress) constant, may not be equal to the address provided for the same block to this instruction’s `indirect labels` operand. The assembly code may only transfer control to addresses provided via this instruction’s `indirect labels`.

On target architectures that implement branch target enforcement by requiring indirect (register-controlled) branch instructions to jump only to locations marked by a special instruction (such as AArch64 `bti`), the called code is expected not to use such an indirect branch to transfer control to the locations in `indirect labels`. Therefore, including a label in the `indirect labels` of a `callbr` does not require the compiler to put a `bti` or equivalent instruction at the label.

##### Arguments:[¶](#id59 "Link to this heading")

This instruction requires several arguments:

1.  The optional “cconv” marker indicates which [calling convention](#callingconv) the call should use. If none is specified, the call defaults to using C calling conventions.
    
2.  The optional [Parameter Attributes](#paramattrs) list for return values. Only ‘`zeroext`’, ‘`signext`’, ‘`noext`’, and ‘`inreg`’ attributes are valid here.
    
3.  The optional addrspace attribute can be used to indicate the address space of the called function. If it is not specified, the program address space from the [datalayout string](#langref-datalayout) will be used.
    
4.  ‘`ty`’: the type of the call instruction itself which is also the type of the return value. Functions that return no value are marked `void`. The signature is computed based on the return type and argument types.
    
5.  ‘`fnty`’: shall be the signature of the function being called. The argument types must match the types implied by this signature. This is only required if the signature specifies a varargs type.
    
6.  ‘`fnptrval`’: An LLVM value containing a pointer to a function to be called. In most cases, this is a direct function call, but other `callbr`’s are just as possible, calling an arbitrary pointer to function value.
    
7.  ‘`function args`’: argument list whose types match the function signature argument types and parameter attributes. All arguments must be of [first class](#t-firstclass) type. If the function signature indicates the function accepts a variable number of arguments, the extra arguments can be specified.
    
8.  ‘`fallthrough label`’: the label reached when the inline assembly’s execution exits the bottom / the intrinsic call returns.
    
9.  ‘`indirect labels`’: the labels reached when a callee transfers control to a location other than the ‘`fallthrough label`’. Label constraints refer to these destinations.
    
10.  The optional [function attributes](#fnattrs) list.
    
11.  The optional [operand bundles](#opbundles) list.
    

##### Semantics:[¶](#id60 "Link to this heading")

This instruction is designed to operate as a standard ‘`call`’ instruction in most regards. The primary difference is that it establishes an association with additional labels to define where control flow goes after the call.

The output values of a ‘`callbr`’ instruction are available both in the the ‘`fallthrough`’ block, and any ‘`indirect`’ blocks(s).

The only current uses of this are:

1.  implement the “goto” feature of gcc inline assembly where additional labels can be provided as locations for the inline assembly to jump to.
    
2.  support selected intrinsics which manipulate control flow and should be chained to specific terminators, such as ‘`unreachable`’.
    

##### Example:[¶](#id61 "Link to this heading")

; "asm goto" without output constraints.
callbr void asm "", "r,!i"(i32 %x)
            to label %fallthrough \[label %indirect\]

; "asm goto" with output constraints.
<result\> \= callbr i32 asm "", "=r,r,!i"(i32 %x)
            to label %fallthrough \[label %indirect\]

; intrinsic which should be followed by unreachable (the order of the
; blocks after the callbr instruction doesn't matter)
  callbr void @llvm.amdgcn.kill(i1 %c) to label %cont \[label %kill\]
cont:
  ...
kill:
  unreachable

#### [‘`resume`’ Instruction](#id2197)[¶](#resume-instruction "Link to this heading")

##### Syntax:[¶](#id62 "Link to this heading")

resume <type\> <value\>

##### Overview:[¶](#id63 "Link to this heading")

The ‘`resume`’ instruction is a terminator instruction that has no successors.

##### Arguments:[¶](#id64 "Link to this heading")

The ‘`resume`’ instruction requires one argument, which must have the same type as the result of any ‘`landingpad`’ instruction in the same function.

##### Semantics:[¶](#id65 "Link to this heading")

The ‘`resume`’ instruction resumes propagation of an existing (in-flight) exception whose unwinding was interrupted with a [landingpad](#i-landingpad) instruction.

##### Example:[¶](#id66 "Link to this heading")

resume { ptr, i32 } %exn

#### [‘`catchswitch`’ Instruction](#id2198)[¶](#catchswitch-instruction "Link to this heading")

##### Syntax:[¶](#id67 "Link to this heading")

<resultval\> \= catchswitch within <parent\> \[ label <handler1\>, label <handler2\>, ... \] unwind to caller
<resultval\> \= catchswitch within <parent\> \[ label <handler1\>, label <handler2\>, ... \] unwind label <default\>

##### Overview:[¶](#id68 "Link to this heading")

The ‘`catchswitch`’ instruction is used by [LLVM’s exception handling system](https://llvm.org/docs/ExceptionHandling.html#overview) to describe the set of possible catch handlers that may be executed by the [EH personality routine](#personalityfn).

##### Arguments:[¶](#id69 "Link to this heading")

The `parent` argument is the token of the funclet that contains the `catchswitch` instruction. If the `catchswitch` is not inside a funclet, this operand may be the token `none`.

The `default` argument is the label of another basic block beginning with either a `cleanuppad` or `catchswitch` instruction. This unwind destination must be a legal target with respect to the `parent` links, as described in the [exception handling documentation](https://llvm.org/docs/ExceptionHandling.html#wineh-constraints).

The `handlers` are a nonempty list of successor blocks that each begin with a [catchpad](#i-catchpad) instruction.

##### Semantics:[¶](#id70 "Link to this heading")

Executing this instruction transfers control to one of the successors in `handlers`, if appropriate, or continues to unwind via the unwind label if present.

The `catchswitch` is both a terminator and a “pad” instruction, meaning that it must be both the first non-phi instruction and last instruction in the basic block. Therefore, it must be the only non-phi instruction in the block.

##### Example:[¶](#id71 "Link to this heading")

dispatch1:
  %cs1 = catchswitch within none \[label %handler0, label %handler1\] unwind to caller
dispatch2:
  %cs2 = catchswitch within %parenthandler \[label %handler0\] unwind label %cleanup

#### [‘`catchret`’ Instruction](#id2199)[¶](#catchret-instruction "Link to this heading")

##### Syntax:[¶](#id72 "Link to this heading")

catchret from <token\> to label <normal\>

##### Overview:[¶](#id73 "Link to this heading")

The ‘`catchret`’ instruction is a terminator instruction that has a single successor.

##### Arguments:[¶](#id74 "Link to this heading")

The first argument to a ‘`catchret`’ indicates which `catchpad` it exits. It must be a [catchpad](#i-catchpad). The second argument to a ‘`catchret`’ specifies where control will transfer to next.

##### Semantics:[¶](#id75 "Link to this heading")

The ‘`catchret`’ instruction ends an existing (in-flight) exception whose unwinding was interrupted with a [catchpad](#i-catchpad) instruction. The [personality function](#personalityfn) gets a chance to execute arbitrary code to, for example, destroy the active exception. Control then transfers to `normal`.

The `token` argument must be a token produced by a `catchpad` instruction. If the specified `catchpad` is not the most-recently-entered not-yet-exited funclet pad (as described in the [EH documentation](https://llvm.org/docs/ExceptionHandling.html#wineh-constraints)), the `catchret`’s behavior is undefined.

##### Example:[¶](#id76 "Link to this heading")

catchret from %catch to label %continue

#### [‘`cleanupret`’ Instruction](#id2200)[¶](#cleanupret-instruction "Link to this heading")

##### Syntax:[¶](#id77 "Link to this heading")

cleanupret from <value\> unwind label <continue\>
cleanupret from <value\> unwind to caller

##### Overview:[¶](#id78 "Link to this heading")

The ‘`cleanupret`’ instruction is a terminator instruction that has an optional successor.

##### Arguments:[¶](#id79 "Link to this heading")

The ‘`cleanupret`’ instruction requires one argument, which indicates which `cleanuppad` it exits, and must be a [cleanuppad](#i-cleanuppad). If the specified `cleanuppad` is not the most-recently-entered not-yet-exited funclet pad (as described in the [EH documentation](https://llvm.org/docs/ExceptionHandling.html#wineh-constraints)), the `cleanupret`’s behavior is undefined.

The ‘`cleanupret`’ instruction also has an optional successor, `continue`, which must be the label of another basic block beginning with either a `cleanuppad` or `catchswitch` instruction. This unwind destination must be a legal target with respect to the `parent` links, as described in the [exception handling documentation](https://llvm.org/docs/ExceptionHandling.html#wineh-constraints).

##### Semantics:[¶](#id82 "Link to this heading")

The ‘`cleanupret`’ instruction indicates to the [personality function](#personalityfn) that one [cleanuppad](#i-cleanuppad) it transferred control to has ended. It transfers control to `continue` or unwinds out of the function.

##### Example:[¶](#id83 "Link to this heading")

cleanupret from %cleanup unwind to caller
cleanupret from %cleanup unwind label %continue

#### [‘`unreachable`’ Instruction](#id2201)[¶](#unreachable-instruction "Link to this heading")

##### Syntax:[¶](#id84 "Link to this heading")

unreachable

##### Overview:[¶](#id85 "Link to this heading")

The ‘`unreachable`’ instruction has no defined semantics. This instruction is used to inform the optimizer that a particular portion of the code is not reachable. This can be used to indicate that the code after a no-return function cannot be reached, and other facts.

##### Semantics:[¶](#id86 "Link to this heading")

The ‘`unreachable`’ instruction has no defined semantics.

### [Unary Operations](#id2202)[¶](#unary-operations "Link to this heading")

Unary operators require a single operand, execute an operation on it, and produce a single value. The operand might represent multiple data, as is the case with the [vector](#t-vector) data type. The result value has the same type as its operand.

#### [‘`fneg`’ Instruction](#id2203)[¶](#fneg-instruction "Link to this heading")

##### Syntax:[¶](#id87 "Link to this heading")

<result\> \= fneg \[fast\-math flags\]\* <ty\> <op1\>   ; yields ty:result

##### Overview:[¶](#id88 "Link to this heading")

The ‘`fneg`’ instruction returns the negation of its operand.

##### Arguments:[¶](#id89 "Link to this heading")

The argument to the ‘`fneg`’ instruction must be a [floating-point](#t-floating) or [vector](#t-vector) of floating-point values.

##### Semantics:[¶](#id90 "Link to this heading")

The value produced is a copy of the operand with its sign bit flipped. The value is otherwise completely identical; in particular, if the input is a NaN, then the quiet/signaling bit and payload are perfectly preserved.

This instruction can also take any number of [fast-math flags](#fastmath), which are optimization hints to enable otherwise unsafe floating-point optimizations:

##### Example:[¶](#id91 "Link to this heading")

<result> = fneg float %val          ; yields float:result = -%var

### [Binary Operations](#id2204)[¶](#binary-operations "Link to this heading")

Binary operators are used to do most of the computation in a program. They require two operands of the same type, execute an operation on them, and produce a single value. The operands might represent multiple data, as is the case with the [vector](#t-vector) data type. The result value has the same type as its operands.

There are several different binary operators:

#### [‘`add`’ Instruction](#id2205)[¶](#add-instruction "Link to this heading")

##### Syntax:[¶](#id92 "Link to this heading")

<result\> \= add <ty\> <op1\>, <op2\>          ; yields ty:result
<result\> \= add nuw <ty\> <op1\>, <op2\>      ; yields ty:result
<result\> \= add nsw <ty\> <op1\>, <op2\>      ; yields ty:result
<result\> \= add nuw nsw <ty\> <op1\>, <op2\>  ; yields ty:result

##### Overview:[¶](#id93 "Link to this heading")

The ‘`add`’ instruction returns the sum of its two operands.

##### Arguments:[¶](#id94 "Link to this heading")

The two arguments to the ‘`add`’ instruction must be [integer](#t-integer) or [vector](#t-vector) of integer values. Both arguments must have identical types.

##### Semantics:[¶](#id95 "Link to this heading")

The value produced is the integer sum of the two operands.

If the sum has unsigned overflow, the result returned is the mathematical result modulo 2n, where n is the bit width of the result.

Because LLVM integers use a two’s complement representation, this instruction is appropriate for both signed and unsigned integers.

`nuw` and `nsw` stand for “No Unsigned Wrap” and “No Signed Wrap”, respectively. If the `nuw` and/or `nsw` keywords are present, the result value of the `add` is a [poison value](#poisonvalues) if unsigned and/or signed overflow, respectively, occurs.

##### Example:[¶](#id96 "Link to this heading")

<result> = add i32 4, %var          ; yields i32:result = 4 + %var

#### [‘`fadd`’ Instruction](#id2206)[¶](#fadd-instruction "Link to this heading")

##### Syntax:[¶](#id97 "Link to this heading")

<result\> \= fadd \[fast\-math flags\]\* <ty\> <op1\>, <op2\>   ; yields ty:result

##### Overview:[¶](#id98 "Link to this heading")

The ‘`fadd`’ instruction returns the sum of its two operands.

##### Arguments:[¶](#id99 "Link to this heading")

The two arguments to the ‘`fadd`’ instruction must be [floating-point](#t-floating) or [vector](#t-vector) of floating-point values. Both arguments must have identical types.

##### Semantics:[¶](#id100 "Link to this heading")

The value produced is the floating-point sum of the two operands. This instruction is assumed to execute in the default [floating-point environment](#floatenv). This instruction can also take any number of [fast-math flags](#fastmath), which are optimization hints to enable otherwise unsafe floating-point optimizations:

##### Example:[¶](#id101 "Link to this heading")

<result> = fadd float 4.0, %var          ; yields float:result = 4.0 + %var

#### [‘`sub`’ Instruction](#id2207)[¶](#sub-instruction "Link to this heading")

##### Syntax:[¶](#id102 "Link to this heading")

<result\> \= sub <ty\> <op1\>, <op2\>          ; yields ty:result
<result\> \= sub nuw <ty\> <op1\>, <op2\>      ; yields ty:result
<result\> \= sub nsw <ty\> <op1\>, <op2\>      ; yields ty:result
<result\> \= sub nuw nsw <ty\> <op1\>, <op2\>  ; yields ty:result

##### Overview:[¶](#id103 "Link to this heading")

The ‘`sub`’ instruction returns the difference of its two operands.

Note that the ‘`sub`’ instruction is used to represent the ‘`neg`’ instruction present in most other intermediate representations.

##### Arguments:[¶](#id104 "Link to this heading")

The two arguments to the ‘`sub`’ instruction must be [integer](#t-integer) or [vector](#t-vector) of integer values. Both arguments must have identical types.

##### Semantics:[¶](#id105 "Link to this heading")

The value produced is the integer difference of the two operands.

If the difference has unsigned overflow, the result returned is the mathematical result modulo 2n, where n is the bit width of the result.

Because LLVM integers use a two’s complement representation, this instruction is appropriate for both signed and unsigned integers.

`nuw` and `nsw` stand for “No Unsigned Wrap” and “No Signed Wrap”, respectively. If the `nuw` and/or `nsw` keywords are present, the result value of the `sub` is a [poison value](#poisonvalues) if unsigned and/or signed overflow, respectively, occurs.

##### Example:[¶](#id106 "Link to this heading")

<result> = sub i32 4, %var          ; yields i32:result = 4 - %var
<result> = sub i32 0, %val          ; yields i32:result = -%var

#### [‘`fsub`’ Instruction](#id2208)[¶](#fsub-instruction "Link to this heading")

##### Syntax:[¶](#id107 "Link to this heading")

<result\> \= fsub \[fast\-math flags\]\* <ty\> <op1\>, <op2\>   ; yields ty:result

##### Overview:[¶](#id108 "Link to this heading")

The ‘`fsub`’ instruction returns the difference of its two operands.

##### Arguments:[¶](#id109 "Link to this heading")

The two arguments to the ‘`fsub`’ instruction must be [floating-point](#t-floating) or [vector](#t-vector) of floating-point values. Both arguments must have identical types.

##### Semantics:[¶](#id110 "Link to this heading")

The value produced is the floating-point difference of the two operands. This instruction is assumed to execute in the default [floating-point environment](#floatenv). This instruction can also take any number of [fast-math flags](#fastmath), which are optimization hints to enable otherwise unsafe floating-point optimizations:

##### Example:[¶](#id111 "Link to this heading")

<result> = fsub float 4.0, %var           ; yields float:result = 4.0 - %var
<result> = fsub float -0.0, %val          ; yields float:result = -%var

#### [‘`mul`’ Instruction](#id2209)[¶](#mul-instruction "Link to this heading")

##### Syntax:[¶](#id112 "Link to this heading")

<result\> \= mul <ty\> <op1\>, <op2\>          ; yields ty:result
<result\> \= mul nuw <ty\> <op1\>, <op2\>      ; yields ty:result
<result\> \= mul nsw <ty\> <op1\>, <op2\>      ; yields ty:result
<result\> \= mul nuw nsw <ty\> <op1\>, <op2\>  ; yields ty:result

##### Overview:[¶](#id113 "Link to this heading")

The ‘`mul`’ instruction returns the product of its two operands.

##### Arguments:[¶](#id114 "Link to this heading")

The two arguments to the ‘`mul`’ instruction must be [integer](#t-integer) or [vector](#t-vector) of integer values. Both arguments must have identical types.

##### Semantics:[¶](#id115 "Link to this heading")

The value produced is the integer product of the two operands.

If the result of the multiplication has unsigned overflow, the result returned is the mathematical result modulo 2n, where n is the bit width of the result.

Because LLVM integers use a two’s complement representation, and the result is the same width as the operands, this instruction returns the correct result for both signed and unsigned integers. If a full product (e.g., `i32` \* `i32` -> `i64`) is needed, the operands should be sign-extended or zero-extended as appropriate to the width of the full product.

`nuw` and `nsw` stand for “No Unsigned Wrap” and “No Signed Wrap”, respectively. If the `nuw` and/or `nsw` keywords are present, the result value of the `mul` is a [poison value](#poisonvalues) if unsigned and/or signed overflow, respectively, occurs.

##### Example:[¶](#id116 "Link to this heading")

<result> = mul i32 4, %var          ; yields i32:result = 4 \* %var

#### [‘`fmul`’ Instruction](#id2210)[¶](#fmul-instruction "Link to this heading")

##### Syntax:[¶](#id117 "Link to this heading")

<result\> \= fmul \[fast\-math flags\]\* <ty\> <op1\>, <op2\>   ; yields ty:result

##### Overview:[¶](#id118 "Link to this heading")

The ‘`fmul`’ instruction returns the product of its two operands.

##### Arguments:[¶](#id119 "Link to this heading")

The two arguments to the ‘`fmul`’ instruction must be [floating-point](#t-floating) or [vector](#t-vector) of floating-point values. Both arguments must have identical types.

##### Semantics:[¶](#id120 "Link to this heading")

The value produced is the floating-point product of the two operands. This instruction is assumed to execute in the default [floating-point environment](#floatenv). This instruction can also take any number of [fast-math flags](#fastmath), which are optimization hints to enable otherwise unsafe floating-point optimizations:

##### Example:[¶](#id121 "Link to this heading")

<result> = fmul float 4.0, %var          ; yields float:result = 4.0 \* %var

#### [‘`udiv`’ Instruction](#id2211)[¶](#udiv-instruction "Link to this heading")

##### Syntax:[¶](#id122 "Link to this heading")

<result\> \= udiv <ty\> <op1\>, <op2\>         ; yields ty:result
<result\> \= udiv exact <ty\> <op1\>, <op2\>   ; yields ty:result

##### Overview:[¶](#id123 "Link to this heading")

The ‘`udiv`’ instruction returns the quotient of its two operands.

##### Arguments:[¶](#id124 "Link to this heading")

The two arguments to the ‘`udiv`’ instruction must be [integer](#t-integer) or [vector](#t-vector) of integer values. Both arguments must have identical types.

##### Semantics:[¶](#id125 "Link to this heading")

The value produced is the unsigned integer quotient of the two operands.

Note that unsigned integer division and signed integer division are distinct operations; for signed integer division, use ‘`sdiv`’.

Division by zero is undefined behavior. For vectors, if any element of the divisor is zero, the operation has undefined behavior.

If the `exact` keyword is present, the result value of the `udiv` is a [poison value](#poisonvalues) if %op1 is not a multiple of %op2 (as such, “((a udiv exact b) mul b) == a”).

##### Example:[¶](#id126 "Link to this heading")

<result> = udiv i32 4, %var          ; yields i32:result = 4 / %var

#### [‘`sdiv`’ Instruction](#id2212)[¶](#sdiv-instruction "Link to this heading")

##### Syntax:[¶](#id127 "Link to this heading")

<result\> \= sdiv <ty\> <op1\>, <op2\>         ; yields ty:result
<result\> \= sdiv exact <ty\> <op1\>, <op2\>   ; yields ty:result

##### Overview:[¶](#id128 "Link to this heading")

The ‘`sdiv`’ instruction returns the quotient of its two operands.

##### Arguments:[¶](#id129 "Link to this heading")

The two arguments to the ‘`sdiv`’ instruction must be [integer](#t-integer) or [vector](#t-vector) of integer values. Both arguments must have identical types.

##### Semantics:[¶](#id130 "Link to this heading")

The value produced is the signed integer quotient of the two operands rounded towards zero.

Note that signed integer division and unsigned integer division are distinct operations; for unsigned integer division, use ‘`udiv`’.

Division by zero is undefined behavior. For vectors, if any element of the divisor is zero, the operation has undefined behavior. Overflow also leads to undefined behavior; this is a rare case, but can occur, for example, by doing a 32-bit division of -2147483648 by -1.

If the `exact` keyword is present, the result value of the `sdiv` is a [poison value](#poisonvalues) if the result would be rounded.

##### Example:[¶](#id131 "Link to this heading")

<result> = sdiv i32 4, %var          ; yields i32:result = 4 / %var

#### [‘`fdiv`’ Instruction](#id2213)[¶](#fdiv-instruction "Link to this heading")

##### Syntax:[¶](#id132 "Link to this heading")

<result\> \= fdiv \[fast\-math flags\]\* <ty\> <op1\>, <op2\>   ; yields ty:result

##### Overview:[¶](#id133 "Link to this heading")

The ‘`fdiv`’ instruction returns the quotient of its two operands.

##### Arguments:[¶](#id134 "Link to this heading")

The two arguments to the ‘`fdiv`’ instruction must be [floating-point](#t-floating) or [vector](#t-vector) of floating-point values. Both arguments must have identical types.

##### Semantics:[¶](#id135 "Link to this heading")

The value produced is the floating-point quotient of the two operands. This instruction is assumed to execute in the default [floating-point environment](#floatenv). This instruction can also take any number of [fast-math flags](#fastmath), which are optimization hints to enable otherwise unsafe floating-point optimizations:

##### Example:[¶](#id136 "Link to this heading")

<result> = fdiv float 4.0, %var          ; yields float:result = 4.0 / %var

#### [‘`urem`’ Instruction](#id2214)[¶](#urem-instruction "Link to this heading")

##### Syntax:[¶](#id137 "Link to this heading")

<result\> \= urem <ty\> <op1\>, <op2\>   ; yields ty:result

##### Overview:[¶](#id138 "Link to this heading")

The ‘`urem`’ instruction returns the remainder from the unsigned division of its two arguments.

##### Arguments:[¶](#id139 "Link to this heading")

The two arguments to the ‘`urem`’ instruction must be [integer](#t-integer) or [vector](#t-vector) of integer values. Both arguments must have identical types.

##### Semantics:[¶](#id140 "Link to this heading")

This instruction returns the unsigned integer _remainder_ of a division. This instruction always performs an unsigned division to get the remainder.

Note that unsigned integer remainder and signed integer remainder are distinct operations; for signed integer remainder, use ‘`srem`’.

Taking the remainder of a division by zero is undefined behavior. For vectors, if any element of the divisor is zero, the operation has undefined behavior.

##### Example:[¶](#id141 "Link to this heading")

<result> = urem i32 4, %var          ; yields i32:result = 4 % %var

#### [‘`srem`’ Instruction](#id2215)[¶](#srem-instruction "Link to this heading")

##### Syntax:[¶](#id142 "Link to this heading")

<result\> \= srem <ty\> <op1\>, <op2\>   ; yields ty:result

##### Overview:[¶](#id143 "Link to this heading")

The ‘`srem`’ instruction returns the remainder from the signed division of its two operands. This instruction can also take [vector](#t-vector) versions of the values in which case the elements must be integers.

##### Arguments:[¶](#id144 "Link to this heading")

The two arguments to the ‘`srem`’ instruction must be [integer](#t-integer) or [vector](#t-vector) of integer values. Both arguments must have identical types.

##### Semantics:[¶](#id145 "Link to this heading")

This instruction returns the _remainder_ of a division (where the result is either zero or has the same sign as the dividend, `op1`), not the _modulo_ operator (where the result is either zero or has the same sign as the divisor, `op2`) of a value. For more information about the difference, see [The Math Forum](http://mathforum.org/dr.math/problems/anne.4.28.99.html). For a table of how this is implemented in various languages, please see [Wikipedia: modulo operation](http://en.wikipedia.org/wiki/Modulo_operation).

Note that signed integer remainder and unsigned integer remainder are distinct operations; for unsigned integer remainder, use ‘`urem`’.

Taking the remainder of a division by zero is undefined behavior. For vectors, if any element of the divisor is zero, the operation has undefined behavior. Overflow also leads to undefined behavior; this is a rare case, but can occur, for example, by taking the remainder of a 32-bit division of -2147483648 by -1. (The remainder doesn’t actually overflow, but this rule lets srem be implemented using instructions that return both the result of the division and the remainder.)

##### Example:[¶](#id146 "Link to this heading")

<result> = srem i32 4, %var          ; yields i32:result = 4 % %var

#### [‘`frem`’ Instruction](#id2216)[¶](#frem-instruction "Link to this heading")

##### Syntax:[¶](#id147 "Link to this heading")

<result\> \= frem \[fast\-math flags\]\* <ty\> <op1\>, <op2\>   ; yields ty:result

##### Overview:[¶](#id148 "Link to this heading")

The ‘`frem`’ instruction returns the remainder from the division of its two operands.

Note

The instruction is implemented as a call to libm’s ‘`fmod`’ for some targets, and using the instruction may thus require linking libm.

##### Arguments:[¶](#id149 "Link to this heading")

The two arguments to the ‘`frem`’ instruction must be [floating-point](#t-floating) or [vector](#t-vector) of floating-point values. Both arguments must have identical types.

##### Semantics:[¶](#id150 "Link to this heading")

The value produced is the floating-point remainder of the two operands. This is the same output as a libm ‘`fmod`’ function, but without any possibility of setting `errno`. The remainder has the same sign as the dividend. This instruction is assumed to execute in the default [floating-point environment](#floatenv). This instruction can also take any number of [fast-math flags](#fastmath), which are optimization hints to enable otherwise unsafe floating-point optimizations:

##### Example:[¶](#id151 "Link to this heading")

<result> = frem float 4.0, %var          ; yields float:result = 4.0 % %var

### [Bitwise Binary Operations](#id2217)[¶](#bitwise-binary-operations "Link to this heading")

Bitwise binary operators are used to do various forms of bit-twiddling in a program. They are generally very efficient instructions and can commonly be strength reduced from other instructions. They require two operands of the same type, execute an operation on them, and produce a single value. The resulting value is the same type as its operands.

#### [‘`shl`’ Instruction](#id2218)[¶](#shl-instruction "Link to this heading")

##### Syntax:[¶](#id152 "Link to this heading")

<result\> \= shl <ty\> <op1\>, <op2\>           ; yields ty:result
<result\> \= shl nuw <ty\> <op1\>, <op2\>       ; yields ty:result
<result\> \= shl nsw <ty\> <op1\>, <op2\>       ; yields ty:result
<result\> \= shl nuw nsw <ty\> <op1\>, <op2\>   ; yields ty:result

##### Overview:[¶](#id153 "Link to this heading")

The ‘`shl`’ instruction returns the first operand shifted to the left a specified number of bits.

##### Arguments:[¶](#id154 "Link to this heading")

Both arguments to the ‘`shl`’ instruction must be the same [integer](#t-integer) or [vector](#t-vector) of integer type. ‘`op2`’ is treated as an unsigned value.

##### Semantics:[¶](#id155 "Link to this heading")

The value produced is `op1` \* 2op2 mod 2n, where `n` is the width of the result. If `op2` is (statically or dynamically) equal to or larger than the number of bits in `op1`, this instruction returns a [poison value](#poisonvalues). If the arguments are vectors, each vector element of `op1` is shifted by the corresponding shift amount in `op2`.

If the `nuw` keyword is present, then the shift produces a poison value if it shifts out any non-zero bits. If the `nsw` keyword is present, then the shift produces a poison value if it shifts out any bits that disagree with the resultant sign bit.

##### Example:[¶](#id156 "Link to this heading")

<result> = shl i32 4, %var   ; yields i32: 4 << %var
<result> = shl i32 4, 2      ; yields i32: 16
<result> = shl i32 1, 10     ; yields i32: 1024
<result> = shl i32 1, 32     ; undefined
<result> = shl <2 x i32> < i32 1, i32 1>, < i32 1, i32 2>   ; yields: result=<2 x i32> < i32 2, i32 4>

#### [‘`lshr`’ Instruction](#id2219)[¶](#lshr-instruction "Link to this heading")

##### Syntax:[¶](#id157 "Link to this heading")

<result\> \= lshr <ty\> <op1\>, <op2\>         ; yields ty:result
<result\> \= lshr exact <ty\> <op1\>, <op2\>   ; yields ty:result

##### Overview:[¶](#id158 "Link to this heading")

The ‘`lshr`’ instruction (logical shift right) returns the first operand shifted to the right a specified number of bits with zero fill.

##### Arguments:[¶](#id159 "Link to this heading")

Both arguments to the ‘`lshr`’ instruction must be the same [integer](#t-integer) or [vector](#t-vector) of integer type. ‘`op2`’ is treated as an unsigned value.

##### Semantics:[¶](#id160 "Link to this heading")

This instruction always performs a logical shift right operation. The most significant bits of the result will be filled with zero bits after the shift. If `op2` is (statically or dynamically) equal to or larger than the number of bits in `op1`, this instruction returns a [poison value](#poisonvalues). If the arguments are vectors, each vector element of `op1` is shifted by the corresponding shift amount in `op2`.

If the `exact` keyword is present, the result value of the `lshr` is a poison value if any of the bits shifted out are non-zero.

##### Example:[¶](#id161 "Link to this heading")

<result> = lshr i32 4, 1   ; yields i32:result = 2
<result> = lshr i32 4, 2   ; yields i32:result = 1
<result> = lshr i8  4, 3   ; yields i8:result = 0
<result> = lshr i8 -2, 1   ; yields i8:result = 0x7F
<result> = lshr i32 1, 32  ; undefined
<result> = lshr <2 x i32> < i32 -2, i32 4>, < i32 1, i32 2>   ; yields: result=<2 x i32> < i32 0x7FFFFFFF, i32 1>

#### [‘`ashr`’ Instruction](#id2220)[¶](#ashr-instruction "Link to this heading")

##### Syntax:[¶](#id162 "Link to this heading")

<result\> \= ashr <ty\> <op1\>, <op2\>         ; yields ty:result
<result\> \= ashr exact <ty\> <op1\>, <op2\>   ; yields ty:result

##### Overview:[¶](#id163 "Link to this heading")

The ‘`ashr`’ instruction (arithmetic shift right) returns the first operand shifted to the right a specified number of bits with sign extension.

##### Arguments:[¶](#id164 "Link to this heading")

Both arguments to the ‘`ashr`’ instruction must be the same [integer](#t-integer) or [vector](#t-vector) of integer type. ‘`op2`’ is treated as an unsigned value.

##### Semantics:[¶](#id165 "Link to this heading")

This instruction always performs an arithmetic shift right operation, The most significant bits of the result will be filled with the sign bit of `op1`. If `op2` is (statically or dynamically) equal to or larger than the number of bits in `op1`, this instruction returns a [poison value](#poisonvalues). If the arguments are vectors, each vector element of `op1` is shifted by the corresponding shift amount in `op2`.

If the `exact` keyword is present, the result value of the `ashr` is a poison value if any of the bits shifted out are non-zero.

##### Example:[¶](#id166 "Link to this heading")

<result> = ashr i32 4, 1   ; yields i32:result = 2
<result> = ashr i32 4, 2   ; yields i32:result = 1
<result> = ashr i8  4, 3   ; yields i8:result = 0
<result> = ashr i8 -2, 1   ; yields i8:result = -1
<result> = ashr i32 1, 32  ; undefined
<result> = ashr <2 x i32> < i32 -2, i32 4>, < i32 1, i32 3>   ; yields: result=<2 x i32> < i32 -1, i32 0>

#### [‘`and`’ Instruction](#id2221)[¶](#and-instruction "Link to this heading")

##### Syntax:[¶](#id167 "Link to this heading")

<result\> \= and <ty\> <op1\>, <op2\>   ; yields ty:result

##### Overview:[¶](#id168 "Link to this heading")

The ‘`and`’ instruction returns the bitwise logical and of its two operands.

##### Arguments:[¶](#id169 "Link to this heading")

The two arguments to the ‘`and`’ instruction must be [integer](#t-integer) or [vector](#t-vector) of integer values. Both arguments must have identical types.

##### Semantics:[¶](#id170 "Link to this heading")

The truth table used for the ‘`and`’ instruction is:

In0

In1

Out

0

0

0

0

1

0

1

0

0

1

1

1

##### Example:[¶](#id171 "Link to this heading")

<result> = and i32 4, %var         ; yields i32:result = 4 & %var
<result> = and i32 15, 40          ; yields i32:result = 8
<result> = and i32 4, 8            ; yields i32:result = 0

#### [‘`or`’ Instruction](#id2222)[¶](#or-instruction "Link to this heading")

##### Syntax:[¶](#id172 "Link to this heading")

<result\> \= or <ty\> <op1\>, <op2\>   ; yields ty:result
<result\> \= or disjoint <ty\> <op1\>, <op2\>   ; yields ty:result

##### Overview:[¶](#id173 "Link to this heading")

The ‘`or`’ instruction returns the bitwise logical inclusive or of its two operands.

##### Arguments:[¶](#id174 "Link to this heading")

The two arguments to the ‘`or`’ instruction must be [integer](#t-integer) or [vector](#t-vector) of integer values. Both arguments must have identical types.

##### Semantics:[¶](#id175 "Link to this heading")

The truth table used for the ‘`or`’ instruction is:

In0

In1

Out

0

0

0

0

1

1

1

0

1

1

1

1

`disjoint` means that for each bit, that bit is zero in at least one of the inputs. This allows the Or to be treated as an Add since no carry can occur from any bit. If the disjoint keyword is present, the result value of the `or` is a [poison value](#poisonvalues) if both inputs have a one in the same bit position. For vectors, only the element containing the bit is poison.

##### Example:[¶](#id176 "Link to this heading")

<result\> \= or i32 4, %var         ; yields i32:result \= 4 | %var
<result\> \= or i32 15, 40          ; yields i32:result \= 47
<result\> \= or i32 4, 8            ; yields i32:result \= 12

#### [‘`xor`’ Instruction](#id2223)[¶](#xor-instruction "Link to this heading")

##### Syntax:[¶](#id177 "Link to this heading")

<result\> \= xor <ty\> <op1\>, <op2\>   ; yields ty:result

##### Overview:[¶](#id178 "Link to this heading")

The ‘`xor`’ instruction returns the bitwise logical exclusive or of its two operands. The `xor` is used to implement the “one’s complement” operation, which is the “~” operator in C.

##### Arguments:[¶](#id179 "Link to this heading")

The two arguments to the ‘`xor`’ instruction must be [integer](#t-integer) or [vector](#t-vector) of integer values. Both arguments must have identical types.

##### Semantics:[¶](#id180 "Link to this heading")

The truth table used for the ‘`xor`’ instruction is:

In0

In1

Out

0

0

0

0

1

1

1

0

1

1

1

0

##### Example:[¶](#id181 "Link to this heading")

<result> = xor i32 4, %var         ; yields i32:result = 4 ^ %var
<result> = xor i32 15, 40          ; yields i32:result = 39
<result> = xor i32 4, 8            ; yields i32:result = 12
<result> = xor i32 %V, -1          ; yields i32:result = ~%V

### [Vector Operations](#id2224)[¶](#vector-operations "Link to this heading")

LLVM supports several instructions to represent vector operations in a target-independent manner. These instructions cover the element-access and vector-specific operations needed to process vectors effectively. While LLVM does directly support these vector operations, many sophisticated algorithms will want to use target-specific intrinsics to take full advantage of a specific target.

#### [‘`extractelement`’ Instruction](#id2225)[¶](#extractelement-instruction "Link to this heading")

##### Syntax:[¶](#id182 "Link to this heading")

<result\> \= extractelement <n x <ty\>> <val\>, <ty2\> <idx\>  ; yields <ty\>
<result\> \= extractelement <vscale x n x <ty\>> <val\>, <ty2\> <idx\> ; yields <ty\>

##### Overview:[¶](#id183 "Link to this heading")

The ‘`extractelement`’ instruction extracts a single scalar element from a vector at a specified index.

##### Arguments:[¶](#id184 "Link to this heading")

The first operand of an ‘`extractelement`’ instruction is a value of [vector](#t-vector) type. The second operand is an index indicating the position from which to extract the element. The index may be a variable of any integer type, and will be treated as an unsigned integer.

##### Semantics:[¶](#id185 "Link to this heading")

The result is a scalar of the same type as the element type of `val`. Its value is the value at position `idx` of `val`. If `idx` exceeds the length of `val` for a fixed-length vector, the result is a [poison value](#poisonvalues). For a scalable vector, if the value of `idx` exceeds the runtime length of the vector, the result is a [poison value](#poisonvalues).

##### Example:[¶](#id186 "Link to this heading")

<result> = extractelement <4 x i32> %vec, i32 0    ; yields i32

#### [‘`insertelement`’ Instruction](#id2226)[¶](#insertelement-instruction "Link to this heading")

##### Syntax:[¶](#id187 "Link to this heading")

<result\> \= insertelement <n x <ty\>> <val\>, <ty\> <elt\>, <ty2\> <idx\>    ; yields <n x <ty\>>
<result\> \= insertelement <vscale x n x <ty\>> <val\>, <ty\> <elt\>, <ty2\> <idx\> ; yields <vscale x n x <ty\>>

##### Overview:[¶](#id188 "Link to this heading")

The ‘`insertelement`’ instruction inserts a scalar element into a vector at a specified index.

##### Arguments:[¶](#id189 "Link to this heading")

The first operand of an ‘`insertelement`’ instruction is a value of [vector](#t-vector) type. The second operand is a scalar value whose type must equal the element type of the first operand. The third operand is an index indicating the position at which to insert the value. The index may be a variable of any integer type, and will be treated as an unsigned integer.

##### Semantics:[¶](#id190 "Link to this heading")

The result is a vector of the same type as `val`. Its element values are those of `val` except at position `idx`, where it gets the value `elt`. If `idx` exceeds the length of `val` for a fixed-length vector, the result is a [poison value](#poisonvalues). For a scalable vector, if the value of `idx` exceeds the runtime length of the vector, the result is a [poison value](#poisonvalues).

##### Example:[¶](#id191 "Link to this heading")

<result> = insertelement <4 x i32> %vec, i32 1, i32 0    ; yields <4 x i32>

#### [‘`shufflevector`’ Instruction](#id2227)[¶](#shufflevector-instruction "Link to this heading")

##### Syntax:[¶](#id192 "Link to this heading")

<result\> \= shufflevector <n x <ty\>> <v1\>, <n x <ty\>> <v2\>, <m x i32\> <mask\>    ; yields <m x <ty\>>
<result\> \= shufflevector <vscale x n x <ty\>> <v1\>, <vscale x n x <ty\>> v2, <vscale x m x i32\> <mask\>  ; yields <vscale x m x <ty\>>

##### Overview:[¶](#id193 "Link to this heading")

The ‘`shufflevector`’ instruction constructs a permutation of elements from two input vectors, returning a vector with the same element type as the input and length that is the same as the shuffle mask.
