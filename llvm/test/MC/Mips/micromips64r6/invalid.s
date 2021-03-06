# RUN: not llvm-mc %s -triple=mips -show-encoding -mcpu=mips64r6 -mattr=micromips 2>%t1
# RUN: FileCheck %s < %t1

  addiur1sp $7, 260        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addiur1sp $7, 241        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: misaligned immediate operand value
  addiur1sp $8, 240        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  addiur2 $9, $7, -1       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  addiur2 $6, $7, 10       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addius5 $7, 9            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addiusp 1032             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  beqzc16 $9, 20           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  beqzc16 $6, 31           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  beqzc16 $6, 130          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bnezc16 $9, 20           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  bnezc16 $6, 31           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bnezc16 $6, 130          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  lbu16 $9, 8($16)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lbu16 $3, -2($16)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lbu16 $3, -2($16)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lbu16 $16, 8($9)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lhu16 $9, 4($16)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lhu16 $3, 64($16)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lhu16 $3, 64($16)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lhu16 $16, 4($9)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lw16  $9, 8($17)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lw16  $4, 68($17)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lw16  $4, 68($17)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lw16  $17, 8($10)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddiv $32, $4, $5         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddiv $3, $34, $5         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddiv $3, $4, $35         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmod $32, $4, $5         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmod $3, $34, $5         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmod $3, $4, $35         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddivu $32, $4, $5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddivu $3, $34, $5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddivu $3, $4, $35        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmodu $32, $4, $5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmodu $3, $34, $5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmodu $3, $4, $35        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  teq $34, $9, 5           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  teq $8, $35, 6           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  teq $8, $9, 16           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  tge $34, $9, 5           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tge $8, $35, 6           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tge $8, $9, 16           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  tgeu $34, $9, 5          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tgeu $8, $35, 6          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tgeu $8, $9, 16          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  tlt $34, $9, 5           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tlt $8, $35, 6           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tlt $8, $9, 16           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  tltu $34, $9, 5          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tltu $8, $35, 6          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tltu $8, $9, 16          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  tne $34, $9, 5           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tne $8, $35, 6           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tne $8, $9, 16           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  teq $8, $9, $2           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tge $8, $9, $2           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tgeu $8, $9, $2          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tlt $8, $9, $2           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tltu $8, $9, $2          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tne $8, $9, $2           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
