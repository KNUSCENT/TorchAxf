#define MUL_FP32
#define Approx_MUL12_2DA 0
#define Approx_MUL12_2JV 0
#define Approx_MUL8u_GTR 0
#define Approx_MUL8u_18UH 0

#define ADD_FP32
#define LOA_SIZE 0
#define AMA5_SIZE 0
#define ETA1_SIZE 0

#if defined(MUL_FP32)
#define MUL_MANTLEN 23
#define MUL_EXPLEN 8
#elif defined(MUL_FP16)
#define MUL_MANTLEN 10
#define MUL_EXPLEN 5
#elif defined(MUL_FP8)
#define MUL_MANTLEN 3
#define MUL_EXPLEN 4
#elif defined(MUL_BF16)
#define MUL_MANTLEN 7
#define MUL_EXPLEN 8
#elif defined(MUL_TF32)
#define MUL_MANTLEN 10
#define MUL_EXPLEN 8
#elif defined(MUL_DLF16) // DLFLOAT16
#define MUL_MANTLEN 9
#define MUL_EXPLEN 6
#elif defined(MUL_CUSTOM)
#define MUL_MANTLEN 4
#define MUL_EXPLEN 3
#else
#define MUL_FP32
#define MUL_MANTLEN 23
#define MUL_EXPLEN 8
#endif

#if defined(ADD_FP32)
#define ADD_MANTLEN 23
#define ADD_EXPLEN 8
#elif defined(ADD_FP16)
#define ADD_MANTLEN 10
#define ADD_EXPLEN 5
#elif defined(ADD_FP8)
#define ADD_MANTLEN 3
#define ADD_EXPLEN 4
#elif defined(ADD_BF16)
#define ADD_MANTLEN 7
#define ADD_EXPLEN 8
#elif defined(ADD_TF32)
#define ADD_MANTLEN 10
#define ADD_EXPLEN 8
#elif defined(ADD_DLF16) // DLFLOAT16
#define ADD_MANTLEN 9
#define ADD_EXPLEN 6
#elif defined(ADD_CUSTOM)
#define ADD_MANTLEN 4
#define ADD_EXPLEN 3
#else
#define ADD_FP32
#define ADD_MANTLEN 23
#define ADD_EXPLEN 8
#endif

#define FP32_EXPLEN 8
#define FP32_MANTLEN 23
#define GRSLEN 3

typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned long long int uint64_t;
typedef long long int int64_t;

#define BITMASK(bitWidth) ((bitWidth > 0) ? (uint64_t)(-1) >> (sizeof(uint64_t) * 8 - (bitWidth)) : 0)
#define GETBIT(a, n) (((a) >> (n)) & BITMASK(1))
#define SWAP(a, b, type) \
    {                    \
        type temp;       \
        temp = a;        \
        a = b;           \
        b = temp;        \
    }

__device__ int tab32[32] = {
    0, 9, 1, 10, 13, 21, 2, 29,
    11, 14, 16, 18, 22, 25, 3, 30,
    8, 12, 20, 28, 15, 17, 24, 7,
    19, 27, 23, 6, 26, 5, 4, 31};

__device__ int GETLOP(uint32_t value) // log2 function
{
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return (int)tab32[(uint32_t)(value * 0x07C4ACDD) >> 27];
}

typedef union
{
    float f;
    struct
    {
        uint32_t mant_bits : FP32_MANTLEN;
        uint32_t exp_bits : FP32_EXPLEN;
        uint32_t sign_bit : 1;
    } p;
} fp32_t;

typedef union
{
    float f;
    struct
    {
        uint32_t mant_bits : MUL_MANTLEN;
        uint32_t exp_bits : MUL_EXPLEN;
        uint32_t sign_bit : 1;
    } p;
} sfp_mul;

typedef union
{
    float f;
    struct
    {
        uint32_t mant_bits : ADD_MANTLEN;
        uint32_t exp_bits : ADD_EXPLEN;
        uint32_t sign_bit : 1;
    } p;
} sfp_add;

__device__ sfp_add add_float_to_sfp(float a) // sfp_t <= FP32
{
    sfp_add r;
    fp32_t t;
    int32_t exp_bits;

    t.f = a;
    exp_bits = t.p.exp_bits - BITMASK(FP32_EXPLEN - 1) + BITMASK(ADD_EXPLEN - 1);

    r.p.mant_bits = (exp_bits < -1) ? 0 : t.p.mant_bits >> (FP32_MANTLEN - ADD_MANTLEN);
    r.p.exp_bits = (exp_bits < 0) ? 0 : (exp_bits > BITMASK(ADD_EXPLEN)) ? BITMASK(ADD_EXPLEN) : exp_bits;
    r.p.sign_bit = t.p.sign_bit;

    return r;
}

__device__ float add_sfp_to_float(sfp_add a)
{
    fp32_t r;

    if (a.p.exp_bits)
    {
        r.p.mant_bits = a.p.mant_bits << (FP32_MANTLEN - ADD_MANTLEN);
        r.p.exp_bits = a.p.exp_bits + (BITMASK(FP32_EXPLEN - 1) - BITMASK(ADD_EXPLEN - 1)); // fp32 bias - fp16/8 bias(2^(8-1)-1)
        r.p.sign_bit = a.p.sign_bit;
    }
    else
    { // Denormalized Numbers
        r.p.mant_bits = (a.p.mant_bits) ? a.p.mant_bits << (FP32_MANTLEN - GETLOP(a.p.mant_bits)) : 0;
        r.p.exp_bits = (a.p.mant_bits) ? a.p.exp_bits - (ADD_MANTLEN - GETLOP(a.p.mant_bits) - 1) + (BITMASK(FP32_EXPLEN - 1) - BITMASK(ADD_EXPLEN - 1)) : 0;
        r.p.sign_bit = a.p.sign_bit;
    }

    return r.f;
}

// reduced precision multiplier
__device__ sfp_mul mul_float_to_sfp(float a) // sfp_t <= FP32
{

    sfp_mul r;
    fp32_t t;
    int32_t exp_bits;

    t.f = a;

    exp_bits = t.p.exp_bits - BITMASK(FP32_EXPLEN - 1) + BITMASK(MUL_EXPLEN - 1);
    r.p.mant_bits = (exp_bits < -1) ? 0 : t.p.mant_bits >> (FP32_MANTLEN - MUL_MANTLEN);
    r.p.exp_bits = (exp_bits < 0) ? 0 : (exp_bits > BITMASK(MUL_EXPLEN)) ? BITMASK(MUL_EXPLEN) : exp_bits;
    r.p.sign_bit = t.p.sign_bit;
    return r;
}

__device__ float mul_sfp_to_float(sfp_mul a)
{
    fp32_t r;

    if (a.p.exp_bits)
    {
        r.p.mant_bits = a.p.mant_bits << (FP32_MANTLEN - MUL_MANTLEN);
        r.p.exp_bits = a.p.exp_bits + (BITMASK(FP32_EXPLEN - 1) - BITMASK(MUL_EXPLEN - 1)); // fp32 bias - fp16/8 bias(2^(8-1)-1)
        r.p.sign_bit = a.p.sign_bit;
    }
    else
    { // Denormalized Numbers
        r.p.mant_bits = (a.p.mant_bits) ? a.p.mant_bits << (FP32_MANTLEN - GETLOP(a.p.mant_bits)) : 0;
        r.p.exp_bits = (a.p.mant_bits) ? a.p.exp_bits - (MUL_MANTLEN - GETLOP(a.p.mant_bits) - 1) + (BITMASK(FP32_EXPLEN - 1) - BITMASK(MUL_EXPLEN - 1)) : 0;
        r.p.sign_bit = a.p.sign_bit;
    }

    return r.f;
}

__device__ sfp_mul custom_multiplier(sfp_mul a, sfp_mul b)
{
    sfp_mul result;
    result.f = 0;

    int32_t a_exp_bits;
    int32_t b_exp_bits;
    uint64_t a_mant_bits;
    uint64_t b_mant_bits;

    result.p.sign_bit = a.p.sign_bit ^ b.p.sign_bit;

    // exception detection
    if ((a.p.exp_bits == 0 && a.p.mant_bits == 0) || (b.p.exp_bits == 0 && b.p.mant_bits == 0))
    {
        return result;
    }

    int BITMASK_EXPLEN = BITMASK(MUL_EXPLEN);
    int BITMASK_MANTLEN = BITMASK(MUL_MANTLEN);

    // Unpacking
    a_exp_bits = (a.p.exp_bits) ? a.p.exp_bits : a.p.exp_bits - (MUL_MANTLEN - GETLOP(a.p.mant_bits) - 1);
    b_exp_bits = (b.p.exp_bits) ? b.p.exp_bits : b.p.exp_bits - (MUL_MANTLEN - GETLOP(b.p.mant_bits) - 1);
    a_mant_bits = (a.p.exp_bits) ? a.p.mant_bits | ((int32_t)1 << MUL_MANTLEN) : a.p.mant_bits << (MUL_MANTLEN - GETLOP(a.p.mant_bits));
    b_mant_bits = (b.p.exp_bits) ? b.p.mant_bits | ((int32_t)1 << MUL_MANTLEN) : b.p.mant_bits << (MUL_MANTLEN - GETLOP(b.p.mant_bits));

    // 1. calculate the exponent of the product
    int32_t c_exp = a_exp_bits + b_exp_bits - BITMASK(MUL_EXPLEN - 1);

    // 2. multiplication
    uint64_t m_mant = a_mant_bits * b_mant_bits;

    // get GRS
    int G = 0;
    int R = 0;
    int S = 0;

    int32_t n_mant = m_mant >> (MUL_MANTLEN + 1);

    if (m_mant & BITMASK(MUL_MANTLEN - 1))
        S = 1;
    G = GETBIT(m_mant, MUL_MANTLEN);
    R = GETBIT(m_mant, MUL_MANTLEN - 1);

    // normalize 1
    int shift_amt = MUL_MANTLEN - GETLOP(n_mant);
    if (n_mant != 0 && !GETBIT(n_mant, MUL_MANTLEN))
    {
        c_exp = c_exp - shift_amt;
        n_mant = (n_mant << 2) | (G << 1) | R;
        n_mant = n_mant << shift_amt;
        G = GETBIT(n_mant, 1);
        R = GETBIT(n_mant, 0);
        n_mant = n_mant >> 2;
    }

    // normalize 2
    if (c_exp < 0)
    {
        shift_amt = -c_exp;

        if (shift_amt > 63)
            shift_amt = 63;

        n_mant = (n_mant << 2) | (G << 1) | R;

        S = ((n_mant & BITMASK(shift_amt)) != 0) | S;

        n_mant = (shift_amt < 26) ? n_mant >> shift_amt : 0;

        G = GETBIT(n_mant, 1);
        R = GETBIT(n_mant, 0);
        n_mant = n_mant >> 2;

        c_exp = 0;
    }

    // 4. round
    int32_t r_mant;
    if (G == 1 && R == 0 && S == 0)
        r_mant = (n_mant) + GETBIT(n_mant, 0);
    else
        r_mant = (n_mant) + (G & (R | S));

    // 5. if need round up
    if (r_mant > BITMASK_MANTLEN)
        c_exp++;
    if (r_mant > BITMASK(MUL_MANTLEN + 1))
        c_exp++;

    // 5. determine the sign and packing
    if (c_exp >= BITMASK_EXPLEN)
    {
        result.p.exp_bits = BITMASK_EXPLEN;
        result.p.mant_bits = 0;
        return result;
    }

    result.p.exp_bits = c_exp;
    result.p.mant_bits = (result.p.exp_bits >= BITMASK_EXPLEN) ? 0 : r_mant & BITMASK_MANTLEN;
    return result;
}

__device__ uint32_t mul12u_2JV(uint16_t A, uint16_t B)
{
  uint32_t P, P_;
  uint16_t tmp, C_10_10,C_10_8,C_10_9,C_11_10,C_11_7,C_11_8,C_11_9,C_8_10,C_9_10,C_9_9,S_10_10,S_10_11,S_10_8,S_10_9,S_11_10,S_11_11,S_11_7,S_11_8,S_11_9,S_12_10,S_12_11,S_12_6,S_12_7,S_12_8,S_12_9,S_7_11,S_8_10,S_8_11,S_9_10,S_9_11,S_9_9;
  S_7_11 = (((A>>7)&1) & ((B>>11)&1));
  S_8_10 = S_7_11^(((A>>8)&1) & ((B>>10)&1));
  C_8_10 = S_7_11&(((A>>8)&1) & ((B>>10)&1));
  S_8_11 = (((A>>8)&1) & ((B>>11)&1));
  S_9_9 = S_8_10^(((A>>9)&1) & ((B>>9)&1));
  C_9_9 = S_8_10&(((A>>9)&1) & ((B>>9)&1));
  tmp = S_8_11^C_8_10;
  S_9_10 = tmp^(((A>>9)&1) & ((B>>10)&1));
  C_9_10 = (tmp&(((A>>9)&1) & ((B>>10)&1)))|(S_8_11&C_8_10);
  S_9_11 = (((A>>9)&1) & ((B>>11)&1));
  S_10_8 = S_9_9^(((A>>10)&1) & ((B>>8)&1));
  C_10_8 = S_9_9&(((A>>10)&1) & ((B>>8)&1));
  tmp = S_9_10^C_9_9;
  S_10_9 = tmp^(((A>>10)&1) & ((B>>9)&1));
  C_10_9 = (tmp&(((A>>10)&1) & ((B>>9)&1)))|(S_9_10&C_9_9);
  tmp = S_9_11^C_9_10;
  S_10_10 = tmp^(((A>>10)&1) & ((B>>10)&1));
  C_10_10 = (tmp&(((A>>10)&1) & ((B>>10)&1)))|(S_9_11&C_9_10);
  S_10_11 = (((A>>10)&1) & ((B>>11)&1));
  S_11_7 = S_10_8^(((A>>11)&1) & ((B>>7)&1));
  C_11_7 = S_10_8&(((A>>11)&1) & ((B>>7)&1));
  tmp = S_10_9^C_10_8;
  S_11_8 = tmp^(((A>>11)&1) & ((B>>8)&1));
  C_11_8 = (tmp&(((A>>11)&1) & ((B>>8)&1)))|(S_10_9&C_10_8);
  tmp = S_10_10^C_10_9;
  S_11_9 = tmp^(((A>>11)&1) & ((B>>9)&1));
  C_11_9 = (tmp&(((A>>11)&1) & ((B>>9)&1)))|(S_10_10&C_10_9);
  tmp = S_10_11^C_10_10;
  S_11_10 = tmp^(((A>>11)&1) & ((B>>10)&1));
  C_11_10 = (tmp&(((A>>11)&1) & ((B>>10)&1)))|(S_10_11&C_10_10);
  S_11_11 = (((A>>11)&1) & ((B>>11)&1));
  P_ = (((C_11_7 & 1)<<1)|((C_11_8 & 1)<<2)|((C_11_9 & 1)<<3)|((C_11_10 & 1)<<4)) + (((S_11_7 & 1)<<0)|((S_11_8 & 1)<<1)|((S_11_9 & 1)<<2)|((S_11_10 & 1)<<3)|((S_11_11 & 1)<<4));
  S_12_6 = (P_ >> 0) & 1;
  S_12_7 = (P_ >> 1) & 1;
  S_12_8 = (P_ >> 2) & 1;
  S_12_9 = (P_ >> 3) & 1;
  S_12_10 = (P_ >> 4) & 1;
  S_12_11 = (P_ >> 5) & 1;
  P = 0;
  P |= (S_12_6 & 1) << 18;
  P |= (S_12_7 & 1) << 19;
  P |= (S_12_8 & 1) << 20;
  P |= (S_12_9 & 1) << 21;
  P |= (S_12_10 & 1) << 22;
  P |= (S_12_11 & 1) << 23;
  return P;
}

__device__ sfp_mul custom_multiplier_2JV(sfp_mul a, sfp_mul b)
{
    sfp_mul result;
    result.f = 0;

    int32_t a_exp_bits;
    int32_t b_exp_bits;
    uint64_t a_mant_bits;
    uint64_t b_mant_bits;

    result.p.sign_bit = a.p.sign_bit ^ b.p.sign_bit;

    // exception detection
    if ((a.p.exp_bits == 0 && a.p.mant_bits == 0) || (b.p.exp_bits == 0 && b.p.mant_bits == 0))
    {
        return result;
    }

    int BITMASK_EXPLEN = BITMASK(MUL_EXPLEN);
    int BITMASK_MANTLEN = BITMASK(MUL_MANTLEN);

    // Unpacking
    a_exp_bits = (a.p.exp_bits) ? a.p.exp_bits : a.p.exp_bits - (MUL_MANTLEN - GETLOP(a.p.mant_bits) - 1);
    b_exp_bits = (b.p.exp_bits) ? b.p.exp_bits : b.p.exp_bits - (MUL_MANTLEN - GETLOP(b.p.mant_bits) - 1);
    a_mant_bits = (a.p.exp_bits) ? a.p.mant_bits | ((int32_t)1 << MUL_MANTLEN) : a.p.mant_bits << (MUL_MANTLEN - GETLOP(a.p.mant_bits));
    b_mant_bits = (b.p.exp_bits) ? b.p.mant_bits | ((int32_t)1 << MUL_MANTLEN) : b.p.mant_bits << (MUL_MANTLEN - GETLOP(b.p.mant_bits));

    // 1. calculate the exponent of the product
    int32_t c_exp = a_exp_bits + b_exp_bits - BITMASK(MUL_EXPLEN - 1);

    // 2. multiplication
    //uint64_t m_mant = a_mant_bits * b_mant_bits;

    long long m_mant;
    uint32_t M1, M2, M3, M4;
	uint16_t AH, AL, BH, BL;

    AH = a_mant_bits >> 12;
	BH = b_mant_bits >> 12;
	AL = a_mant_bits & BITMASK(12);
	BL = b_mant_bits & BITMASK(12);

    M1 = mul12u_2JV(AL, BL);
	M2 = mul12u_2JV(AH, BL);
	M3 = mul12u_2JV(AL, BH);
	M4 = mul12u_2JV(AH, BH);

    m_mant = ((long long)M4 << 24) + ((long long)M3 << 12) + ((long long)M2 << 12) + (long long)M1;

    int32_t n_mant = m_mant >> (MUL_MANTLEN + 1);

    // normalize 1
    int shift_amt = MUL_MANTLEN - GETLOP(n_mant);
    if (n_mant != 0 && !GETBIT(n_mant, MUL_MANTLEN))
    {
        c_exp = c_exp - shift_amt;
        n_mant = n_mant << shift_amt;
    }

    // normalize 2
    if (c_exp < 0)
    {
        shift_amt = -c_exp;
        n_mant = (shift_amt < 24) ? n_mant >> shift_amt : 0;
        c_exp = 0;
    }

    // 4. round
    int32_t r_mant = n_mant;

    // 5. if need round up
    if (r_mant > BITMASK_MANTLEN)
        c_exp++;
    if (r_mant > BITMASK(MUL_MANTLEN + 1))
        c_exp++;

    // 5. determine the sign and packing
    if (c_exp >= BITMASK_EXPLEN)
    {
        result.p.exp_bits = BITMASK_EXPLEN;
        result.p.mant_bits = 0;
        return result;
    }

    result.p.exp_bits = c_exp;
    result.p.mant_bits = (result.p.exp_bits >= BITMASK_EXPLEN) ? 0 : r_mant & BITMASK_MANTLEN;
    return result;
}

__device__ uint32_t mul12u_2DA(uint16_t A, uint16_t B)
{
  uint32_t P, P_;
  uint16_t tmp, C_10_10,C_10_2,C_10_3,C_10_4,C_10_5,C_10_6,C_10_7,C_10_8,C_10_9,C_11_1,C_11_10,C_11_2,C_11_3,C_11_4,C_11_5,C_11_6,C_11_7,C_11_8,C_11_9,C_12_1,C_12_10,C_12_2,C_12_3,C_12_4,C_12_5,C_12_6,C_12_7,C_12_8,C_12_9,C_3_10,C_3_9,C_4_10,C_4_8,C_4_9,C_5_10,C_5_7,C_5_8,C_5_9,C_6_10,C_6_6,C_6_7,C_6_8,C_6_9,C_7_10,C_7_5,C_7_6,C_7_7,C_7_8,C_7_9,C_8_10,C_8_4,C_8_5,C_8_6,C_8_7,C_8_8,C_8_9,C_9_10,C_9_3,C_9_4,C_9_5,C_9_6,C_9_7,C_9_8,C_9_9,S_10_10,S_10_11,S_10_2,S_10_3,S_10_4,S_10_5,S_10_6,S_10_7,S_10_8,S_10_9,S_11_1,S_11_10,S_11_11,S_11_2,S_11_3,S_11_4,S_11_5,S_11_6,S_11_7,S_11_8,S_11_9,S_12_0,S_12_1,S_12_10,S_12_11,S_12_2,S_12_3,S_12_4,S_12_5,S_12_6,S_12_7,S_12_8,S_12_9,S_2_10,S_2_11,S_3_10,S_3_11,S_3_9,S_4_10,S_4_11,S_4_8,S_4_9,S_5_10,S_5_11,S_5_7,S_5_8,S_5_9,S_6_10,S_6_11,S_6_6,S_6_7,S_6_8,S_6_9,S_7_10,S_7_11,S_7_5,S_7_6,S_7_7,S_7_8,S_7_9,S_8_10,S_8_11,S_8_4,S_8_5,S_8_6,S_8_7,S_8_8,S_8_9,S_9_10,S_9_11,S_9_3,S_9_4,S_9_5,S_9_6,S_9_7,S_9_8,S_9_9;
  S_2_10 = (((A>>2)&1) & ((B>>10)&1));
  S_2_11 = (((A>>2)&1) & ((B>>11)&1));
  S_3_9 = S_2_10^(((A>>3)&1) & ((B>>9)&1));
  C_3_9 = S_2_10&(((A>>3)&1) & ((B>>9)&1));
  S_3_10 = S_2_11^(((A>>3)&1) & ((B>>10)&1));
  C_3_10 = S_2_11&(((A>>3)&1) & ((B>>10)&1));
  S_3_11 = (((A>>3)&1) & ((B>>11)&1));
  S_4_8 = S_3_9^(((A>>4)&1) & ((B>>8)&1));
  C_4_8 = S_3_9&(((A>>4)&1) & ((B>>8)&1));
  tmp = S_3_10^C_3_9;
  S_4_9 = tmp^(((A>>4)&1) & ((B>>9)&1));
  C_4_9 = (tmp&(((A>>4)&1) & ((B>>9)&1)))|(S_3_10&C_3_9);
  tmp = S_3_11^C_3_10;
  S_4_10 = tmp^(((A>>4)&1) & ((B>>10)&1));
  C_4_10 = (tmp&(((A>>4)&1) & ((B>>10)&1)))|(S_3_11&C_3_10);
  S_4_11 = (((A>>4)&1) & ((B>>11)&1));
  S_5_7 = S_4_8^(((A>>5)&1) & ((B>>7)&1));
  C_5_7 = S_4_8&(((A>>5)&1) & ((B>>7)&1));
  tmp = S_4_9^C_4_8;
  S_5_8 = tmp^(((A>>5)&1) & ((B>>8)&1));
  C_5_8 = (tmp&(((A>>5)&1) & ((B>>8)&1)))|(S_4_9&C_4_8);
  tmp = S_4_10^C_4_9;
  S_5_9 = tmp^(((A>>5)&1) & ((B>>9)&1));
  C_5_9 = (tmp&(((A>>5)&1) & ((B>>9)&1)))|(S_4_10&C_4_9);
  tmp = S_4_11^C_4_10;
  S_5_10 = tmp^(((A>>5)&1) & ((B>>10)&1));
  C_5_10 = (tmp&(((A>>5)&1) & ((B>>10)&1)))|(S_4_11&C_4_10);
  S_5_11 = (((A>>5)&1) & ((B>>11)&1));
  S_6_6 = S_5_7^(((A>>6)&1) & ((B>>6)&1));
  C_6_6 = S_5_7&(((A>>6)&1) & ((B>>6)&1));
  tmp = S_5_8^C_5_7;
  S_6_7 = tmp^(((A>>6)&1) & ((B>>7)&1));
  C_6_7 = (tmp&(((A>>6)&1) & ((B>>7)&1)))|(S_5_8&C_5_7);
  tmp = S_5_9^C_5_8;
  S_6_8 = tmp^(((A>>6)&1) & ((B>>8)&1));
  C_6_8 = (tmp&(((A>>6)&1) & ((B>>8)&1)))|(S_5_9&C_5_8);
  tmp = S_5_10^C_5_9;
  S_6_9 = tmp^(((A>>6)&1) & ((B>>9)&1));
  C_6_9 = (tmp&(((A>>6)&1) & ((B>>9)&1)))|(S_5_10&C_5_9);
  tmp = S_5_11^C_5_10;
  S_6_10 = tmp^(((A>>6)&1) & ((B>>10)&1));
  C_6_10 = (tmp&(((A>>6)&1) & ((B>>10)&1)))|(S_5_11&C_5_10);
  S_6_11 = (((A>>6)&1) & ((B>>11)&1));
  S_7_5 = S_6_6^(((A>>7)&1) & ((B>>5)&1));
  C_7_5 = S_6_6&(((A>>7)&1) & ((B>>5)&1));
  tmp = S_6_7^C_6_6;
  S_7_6 = tmp^(((A>>7)&1) & ((B>>6)&1));
  C_7_6 = (tmp&(((A>>7)&1) & ((B>>6)&1)))|(S_6_7&C_6_6);
  tmp = S_6_8^C_6_7;
  S_7_7 = tmp^(((A>>7)&1) & ((B>>7)&1));
  C_7_7 = (tmp&(((A>>7)&1) & ((B>>7)&1)))|(S_6_8&C_6_7);
  tmp = S_6_9^C_6_8;
  S_7_8 = tmp^(((A>>7)&1) & ((B>>8)&1));
  C_7_8 = (tmp&(((A>>7)&1) & ((B>>8)&1)))|(S_6_9&C_6_8);
  tmp = S_6_10^C_6_9;
  S_7_9 = tmp^(((A>>7)&1) & ((B>>9)&1));
  C_7_9 = (tmp&(((A>>7)&1) & ((B>>9)&1)))|(S_6_10&C_6_9);
  tmp = S_6_11^C_6_10;
  S_7_10 = tmp^(((A>>7)&1) & ((B>>10)&1));
  C_7_10 = (tmp&(((A>>7)&1) & ((B>>10)&1)))|(S_6_11&C_6_10);
  S_7_11 = (((A>>7)&1) & ((B>>11)&1));
  S_8_4 = S_7_5^(((A>>8)&1) & ((B>>4)&1));
  C_8_4 = S_7_5&(((A>>8)&1) & ((B>>4)&1));
  tmp = S_7_6^C_7_5;
  S_8_5 = tmp^(((A>>8)&1) & ((B>>5)&1));
  C_8_5 = (tmp&(((A>>8)&1) & ((B>>5)&1)))|(S_7_6&C_7_5);
  tmp = S_7_7^C_7_6;
  S_8_6 = tmp^(((A>>8)&1) & ((B>>6)&1));
  C_8_6 = (tmp&(((A>>8)&1) & ((B>>6)&1)))|(S_7_7&C_7_6);
  tmp = S_7_8^C_7_7;
  S_8_7 = tmp^(((A>>8)&1) & ((B>>7)&1));
  C_8_7 = (tmp&(((A>>8)&1) & ((B>>7)&1)))|(S_7_8&C_7_7);
  tmp = S_7_9^C_7_8;
  S_8_8 = tmp^(((A>>8)&1) & ((B>>8)&1));
  C_8_8 = (tmp&(((A>>8)&1) & ((B>>8)&1)))|(S_7_9&C_7_8);
  tmp = S_7_10^C_7_9;
  S_8_9 = tmp^(((A>>8)&1) & ((B>>9)&1));
  C_8_9 = (tmp&(((A>>8)&1) & ((B>>9)&1)))|(S_7_10&C_7_9);
  tmp = S_7_11^C_7_10;
  S_8_10 = tmp^(((A>>8)&1) & ((B>>10)&1));
  C_8_10 = (tmp&(((A>>8)&1) & ((B>>10)&1)))|(S_7_11&C_7_10);
  S_8_11 = (((A>>8)&1) & ((B>>11)&1));
  S_9_3 = S_8_4^(((A>>9)&1) & ((B>>3)&1));
  C_9_3 = S_8_4&(((A>>9)&1) & ((B>>3)&1));
  tmp = S_8_5^C_8_4;
  S_9_4 = tmp^(((A>>9)&1) & ((B>>4)&1));
  C_9_4 = (tmp&(((A>>9)&1) & ((B>>4)&1)))|(S_8_5&C_8_4);
  tmp = S_8_6^C_8_5;
  S_9_5 = tmp^(((A>>9)&1) & ((B>>5)&1));
  C_9_5 = (tmp&(((A>>9)&1) & ((B>>5)&1)))|(S_8_6&C_8_5);
  tmp = S_8_7^C_8_6;
  S_9_6 = tmp^(((A>>9)&1) & ((B>>6)&1));
  C_9_6 = (tmp&(((A>>9)&1) & ((B>>6)&1)))|(S_8_7&C_8_6);
  tmp = S_8_8^C_8_7;
  S_9_7 = tmp^(((A>>9)&1) & ((B>>7)&1));
  C_9_7 = (tmp&(((A>>9)&1) & ((B>>7)&1)))|(S_8_8&C_8_7);
  tmp = S_8_9^C_8_8;
  S_9_8 = tmp^(((A>>9)&1) & ((B>>8)&1));
  C_9_8 = (tmp&(((A>>9)&1) & ((B>>8)&1)))|(S_8_9&C_8_8);
  tmp = S_8_10^C_8_9;
  S_9_9 = tmp^(((A>>9)&1) & ((B>>9)&1));
  C_9_9 = (tmp&(((A>>9)&1) & ((B>>9)&1)))|(S_8_10&C_8_9);
  tmp = S_8_11^C_8_10;
  S_9_10 = tmp^(((A>>9)&1) & ((B>>10)&1));
  C_9_10 = (tmp&(((A>>9)&1) & ((B>>10)&1)))|(S_8_11&C_8_10);
  S_9_11 = (((A>>9)&1) & ((B>>11)&1));
  S_10_2 = S_9_3^(((A>>10)&1) & ((B>>2)&1));
  C_10_2 = S_9_3&(((A>>10)&1) & ((B>>2)&1));
  tmp = S_9_4^C_9_3;
  S_10_3 = tmp^(((A>>10)&1) & ((B>>3)&1));
  C_10_3 = (tmp&(((A>>10)&1) & ((B>>3)&1)))|(S_9_4&C_9_3);
  tmp = S_9_5^C_9_4;
  S_10_4 = tmp^(((A>>10)&1) & ((B>>4)&1));
  C_10_4 = (tmp&(((A>>10)&1) & ((B>>4)&1)))|(S_9_5&C_9_4);
  tmp = S_9_6^C_9_5;
  S_10_5 = tmp^(((A>>10)&1) & ((B>>5)&1));
  C_10_5 = (tmp&(((A>>10)&1) & ((B>>5)&1)))|(S_9_6&C_9_5);
  tmp = S_9_7^C_9_6;
  S_10_6 = tmp^(((A>>10)&1) & ((B>>6)&1));
  C_10_6 = (tmp&(((A>>10)&1) & ((B>>6)&1)))|(S_9_7&C_9_6);
  tmp = S_9_8^C_9_7;
  S_10_7 = tmp^(((A>>10)&1) & ((B>>7)&1));
  C_10_7 = (tmp&(((A>>10)&1) & ((B>>7)&1)))|(S_9_8&C_9_7);
  tmp = S_9_9^C_9_8;
  S_10_8 = tmp^(((A>>10)&1) & ((B>>8)&1));
  C_10_8 = (tmp&(((A>>10)&1) & ((B>>8)&1)))|(S_9_9&C_9_8);
  tmp = S_9_10^C_9_9;
  S_10_9 = tmp^(((A>>10)&1) & ((B>>9)&1));
  C_10_9 = (tmp&(((A>>10)&1) & ((B>>9)&1)))|(S_9_10&C_9_9);
  tmp = S_9_11^C_9_10;
  S_10_10 = tmp^(((A>>10)&1) & ((B>>10)&1));
  C_10_10 = (tmp&(((A>>10)&1) & ((B>>10)&1)))|(S_9_11&C_9_10);
  S_10_11 = (((A>>10)&1) & ((B>>11)&1));
  S_11_1 = S_10_2^(((A>>11)&1) & ((B>>1)&1));
  C_11_1 = S_10_2&(((A>>11)&1) & ((B>>1)&1));
  tmp = S_10_3^C_10_2;
  S_11_2 = tmp^(((A>>11)&1) & ((B>>2)&1));
  C_11_2 = (tmp&(((A>>11)&1) & ((B>>2)&1)))|(S_10_3&C_10_2);
  tmp = S_10_4^C_10_3;
  S_11_3 = tmp^(((A>>11)&1) & ((B>>3)&1));
  C_11_3 = (tmp&(((A>>11)&1) & ((B>>3)&1)))|(S_10_4&C_10_3);
  tmp = S_10_5^C_10_4;
  S_11_4 = tmp^(((A>>11)&1) & ((B>>4)&1));
  C_11_4 = (tmp&(((A>>11)&1) & ((B>>4)&1)))|(S_10_5&C_10_4);
  tmp = S_10_6^C_10_5;
  S_11_5 = tmp^(((A>>11)&1) & ((B>>5)&1));
  C_11_5 = (tmp&(((A>>11)&1) & ((B>>5)&1)))|(S_10_6&C_10_5);
  tmp = S_10_7^C_10_6;
  S_11_6 = tmp^(((A>>11)&1) & ((B>>6)&1));
  C_11_6 = (tmp&(((A>>11)&1) & ((B>>6)&1)))|(S_10_7&C_10_6);
  tmp = S_10_8^C_10_7;
  S_11_7 = tmp^(((A>>11)&1) & ((B>>7)&1));
  C_11_7 = (tmp&(((A>>11)&1) & ((B>>7)&1)))|(S_10_8&C_10_7);
  tmp = S_10_9^C_10_8;
  S_11_8 = tmp^(((A>>11)&1) & ((B>>8)&1));
  C_11_8 = (tmp&(((A>>11)&1) & ((B>>8)&1)))|(S_10_9&C_10_8);
  tmp = S_10_10^C_10_9;
  S_11_9 = tmp^(((A>>11)&1) & ((B>>9)&1));
  C_11_9 = (tmp&(((A>>11)&1) & ((B>>9)&1)))|(S_10_10&C_10_9);
  tmp = S_10_11^C_10_10;
  S_11_10 = tmp^(((A>>11)&1) & ((B>>10)&1));
  C_11_10 = (tmp&(((A>>11)&1) & ((B>>10)&1)))|(S_10_11&C_10_10);
  S_11_11 = (((A>>11)&1) & ((B>>11)&1));
  S_12_0 = S_11_1;
  S_12_1 = S_11_2^C_11_1;
  C_12_1 = S_11_2&C_11_1;
  tmp = S_11_3^C_12_1;
  S_12_2 = tmp^C_11_2;
  C_12_2 = (tmp&C_11_2)|(S_11_3&C_12_1);
  tmp = S_11_4^C_12_2;
  S_12_3 = tmp^C_11_3;
  C_12_3 = (tmp&C_11_3)|(S_11_4&C_12_2);
  tmp = S_11_5^C_12_3;
  S_12_4 = tmp^C_11_4;
  C_12_4 = (tmp&C_11_4)|(S_11_5&C_12_3);
  tmp = S_11_6^C_12_4;
  S_12_5 = tmp^C_11_5;
  C_12_5 = (tmp&C_11_5)|(S_11_6&C_12_4);
  tmp = S_11_7^C_12_5;
  S_12_6 = tmp^C_11_6;
  C_12_6 = (tmp&C_11_6)|(S_11_7&C_12_5);
  tmp = S_11_8^C_12_6;
  S_12_7 = tmp^C_11_7;
  C_12_7 = (tmp&C_11_7)|(S_11_8&C_12_6);
  tmp = S_11_9^C_12_7;
  S_12_8 = tmp^C_11_8;
  C_12_8 = (tmp&C_11_8)|(S_11_9&C_12_7);
  tmp = S_11_10^C_12_8;
  S_12_9 = tmp^C_11_9;
  C_12_9 = (tmp&C_11_9)|(S_11_10&C_12_8);
  tmp = S_11_11^C_12_9;
  S_12_10 = tmp^C_11_10;
  C_12_10 = (tmp&C_11_10)|(S_11_11&C_12_9);
  S_12_11 = C_12_10;
  P = 0;
  P |= (S_12_0 & 1) << 12;
  P |= (S_12_1 & 1) << 13;
  P |= (S_12_2 & 1) << 14;
  P |= (S_12_3 & 1) << 15;
  P |= (S_12_4 & 1) << 16;
  P |= (S_12_5 & 1) << 17;
  P |= (S_12_6 & 1) << 18;
  P |= (S_12_7 & 1) << 19;
  P |= (S_12_8 & 1) << 20;
  P |= (S_12_9 & 1) << 21;
  P |= (S_12_10 & 1) << 22;
  P |= (S_12_11 & 1) << 23;
  return P;
}

__device__ sfp_mul custom_multiplier_2DA(sfp_mul a, sfp_mul b)
{
    sfp_mul result;
    result.f = 0;

    int32_t a_exp_bits;
    int32_t b_exp_bits;
    uint64_t a_mant_bits;
    uint64_t b_mant_bits;

    result.p.sign_bit = a.p.sign_bit ^ b.p.sign_bit;

    // exception detection
    if ((a.p.exp_bits == 0 && a.p.mant_bits == 0) || (b.p.exp_bits == 0 && b.p.mant_bits == 0))
    {
        return result;
    }

    int BITMASK_EXPLEN = BITMASK(MUL_EXPLEN);
    int BITMASK_MANTLEN = BITMASK(MUL_MANTLEN);

    // Unpacking
    a_exp_bits = (a.p.exp_bits) ? a.p.exp_bits : a.p.exp_bits - (MUL_MANTLEN - GETLOP(a.p.mant_bits) - 1);
    b_exp_bits = (b.p.exp_bits) ? b.p.exp_bits : b.p.exp_bits - (MUL_MANTLEN - GETLOP(b.p.mant_bits) - 1);
    a_mant_bits = (a.p.exp_bits) ? a.p.mant_bits | ((int32_t)1 << MUL_MANTLEN) : a.p.mant_bits << (MUL_MANTLEN - GETLOP(a.p.mant_bits));
    b_mant_bits = (b.p.exp_bits) ? b.p.mant_bits | ((int32_t)1 << MUL_MANTLEN) : b.p.mant_bits << (MUL_MANTLEN - GETLOP(b.p.mant_bits));

    // 1. calculate the exponent of the product
    int32_t c_exp = a_exp_bits + b_exp_bits - BITMASK(MUL_EXPLEN - 1);

    // 2. multiplication
    //uint64_t m_mant = a_mant_bits * b_mant_bits;

    long long m_mant;
    uint32_t M1, M2, M3, M4;
	uint16_t AH, AL, BH, BL;

    AH = a_mant_bits >> 12;
	BH = b_mant_bits >> 12;
	AL = a_mant_bits & BITMASK(12);
	BL = b_mant_bits & BITMASK(12);

    M1 = mul12u_2DA(AL, BL);
	M2 = mul12u_2DA(AH, BL);
	M3 = mul12u_2DA(AL, BH);
	M4 = mul12u_2DA(AH, BH);

    m_mant = ((long long)M4 << 24) + ((long long)M3 << 12) + ((long long)M2 << 12) + (long long)M1;

    int32_t n_mant = m_mant >> (MUL_MANTLEN + 1);

    // normalize 1
    int shift_amt = MUL_MANTLEN - GETLOP(n_mant);
    if (n_mant != 0 && !GETBIT(n_mant, MUL_MANTLEN))
    {
        c_exp = c_exp - shift_amt;
        n_mant = n_mant << shift_amt;
    }

    // normalize 2
    if (c_exp < 0)
    {
        shift_amt = -c_exp;
        n_mant = (shift_amt < 24) ? n_mant >> shift_amt : 0;
        c_exp = 0;
    }

    // 4. round
    int32_t r_mant = n_mant;

    // 5. if need round up
    if (r_mant > BITMASK_MANTLEN)
        c_exp++;
    if (r_mant > BITMASK(MUL_MANTLEN + 1))
        c_exp++;

    // 5. determine the sign and packing
    if (c_exp >= BITMASK_EXPLEN)
    {
        result.p.exp_bits = BITMASK_EXPLEN;
        result.p.mant_bits = 0;
        return result;
    }

    result.p.exp_bits = c_exp;
    result.p.mant_bits = (result.p.exp_bits >= BITMASK_EXPLEN) ? 0 : r_mant & BITMASK_MANTLEN;
    return result;
}

__device__ uint64_t mul8u_GTR(const uint64_t B,const uint64_t A)
{
   uint64_t O, dout_22, dout_23, dout_28, dout_29, dout_30, dout_31, dout_42, dout_43, dout_44, dout_45, dout_50, dout_51, dout_52, dout_53, dout_75, dout_77, dout_79, dout_80, dout_81, dout_82, dout_83, dout_84, dout_85, dout_87, dout_88, dout_89, dout_90, dout_91, dout_92, dout_93, dout_94, dout_95, dout_96, dout_107, dout_112, dout_113, dout_117, dout_118, dout_119, dout_120, dout_121, dout_122, dout_123, dout_124, dout_125, dout_126, dout_127, dout_128, dout_129, dout_130, dout_131, dout_132, dout_133, dout_134, dout_135, dout_136, dout_137, dout_138, dout_139, dout_140, dout_145, dout_146, dout_150, dout_151, dout_153, dout_155, dout_157, dout_158, dout_159, dout_160, dout_161, dout_162, dout_163, dout_164, dout_165, dout_166, dout_167, dout_168, dout_169, dout_170, dout_171, dout_172, dout_173, dout_174, dout_175, dout_176, dout_177, dout_178, dout_179, dout_180, dout_181, dout_182, dout_183, dout_184, dout_188, dout_189, dout_190, dout_191, dout_192, dout_193, dout_194, dout_195, dout_196, dout_197, dout_198, dout_199, dout_200, dout_201, dout_202, dout_203, dout_204, dout_205, dout_206, dout_207, dout_208, dout_209, dout_210, dout_211, dout_212, dout_213, dout_214, dout_215, dout_216, dout_217, dout_218, dout_219, dout_220, dout_221, dout_222, dout_223, dout_224, dout_225, dout_226, dout_227, dout_228, dout_229, dout_230, dout_231, dout_232, dout_233, dout_234, dout_235, dout_236, dout_237, dout_238, dout_239, dout_240, dout_241, dout_242, dout_243, dout_244, dout_245, dout_246, dout_247, dout_248, dout_249, dout_250, dout_251, dout_252, dout_253, dout_254, dout_255, dout_256, dout_257, dout_258, dout_259, dout_260, dout_261, dout_262, dout_263, dout_264, dout_265, dout_266, dout_267, dout_268, dout_269, dout_270, dout_271, dout_272, dout_273, dout_274, dout_275, dout_276, dout_277, dout_278, dout_279, dout_280, dout_281, dout_282, dout_283, dout_284, dout_285, dout_286, dout_287, dout_288, dout_289, dout_290, dout_291, dout_292, dout_293, dout_294, dout_295, dout_296, dout_297, dout_298, dout_299, dout_300, dout_301, dout_302, dout_303, dout_304, dout_305, dout_306, dout_307, dout_308, dout_309, dout_310, dout_311, dout_312, dout_313, dout_314, dout_315, dout_316, dout_317, dout_318, dout_319, dout_320, dout_321, dout_322, dout_323, dout_324, dout_325, dout_326, dout_327, dout_328, dout_329, dout_330, dout_331, dout_332, dout_333, dout_334, dout_335;   int avg=0;

   dout_22=((B >> 6)&1)&((A >> 0)&1);
   dout_23=((B >> 7)&1)&((A >> 0)&1);
   dout_28=((B >> 3)&1)&((A >> 2)&1);
   dout_29=((B >> 5)&1)&((A >> 1)&1);
   dout_30=((B >> 6)&1)&((A >> 1)&1);
   dout_31=((B >> 7)&1)&((A >> 1)&1);
   dout_42=dout_22|dout_29;
   dout_43=dout_22&dout_29;
   dout_44=dout_23^dout_30;
   dout_45=dout_23&dout_30;
   dout_50=((B >> 4)&1)&((A >> 2)&1);
   dout_51=((B >> 5)&1)&((A >> 2)&1);
   dout_52=((B >> 6)&1)&((A >> 2)&1);
   dout_53=((B >> 7)&1)&((A >> 2)&1);
   dout_75=dout_42&dout_50;
   dout_77=dout_42|dout_50;
   dout_79=dout_44^dout_51;
   dout_80=dout_44&dout_51;
   dout_81=dout_79&dout_43;
   dout_82=dout_79^dout_43;
   dout_83=dout_80|dout_81;
   dout_84=dout_31^dout_52;
   dout_85=dout_31&dout_52;
   dout_87=dout_84^dout_45;
   dout_88=dout_85|dout_45;
   dout_89=((B >> 0)&1)&((A >> 3)&1);
   dout_90=((B >> 1)&1)&((A >> 3)&1);
   dout_91=((B >> 2)&1)&((A >> 3)&1);
   dout_92=((B >> 3)&1)&((A >> 3)&1);
   dout_93=((B >> 4)&1)&((A >> 3)&1);
   dout_94=((B >> 5)&1)&((A >> 3)&1);
   dout_95=((B >> 6)&1)&((A >> 3)&1);
   dout_96=((B >> 7)&1)&((A >> 3)&1);
   dout_107=dout_28|dout_91;
   dout_112=dout_77|dout_92;
   dout_113=dout_77&dout_92;
   dout_117=dout_82^dout_93;
   dout_118=dout_82&dout_93;
   dout_119=dout_117&dout_75;
   dout_120=dout_117^dout_75;
   dout_121=dout_118|dout_119;
   dout_122=dout_87^dout_94;
   dout_123=dout_87&dout_94;
   dout_124=dout_122&dout_83;
   dout_125=dout_122^dout_83;
   dout_126=dout_123|dout_124;
   dout_127=dout_53^dout_95;
   dout_128=dout_53&dout_95;
   dout_129=dout_127&dout_88;
   dout_130=dout_127^dout_88;
   dout_131=dout_128|dout_129;
   dout_132=((B >> 0)&1)&((A >> 4)&1);
   dout_133=((B >> 1)&1)&((A >> 4)&1);
   dout_134=((B >> 2)&1)&((A >> 4)&1);
   dout_135=((B >> 3)&1)&((A >> 4)&1);
   dout_136=((B >> 4)&1)&((A >> 4)&1);
   dout_137=((B >> 5)&1)&((A >> 4)&1);
   dout_138=((B >> 6)&1)&((A >> 4)&1);
   dout_139=((B >> 7)&1)&((A >> 4)&1);
   dout_140=dout_90|dout_132;
   dout_145=dout_107^dout_133;
   dout_146=dout_107&dout_133;
   dout_150=dout_112^dout_134;
   dout_151=dout_112&dout_134;
   dout_153=dout_135&dout_120;
   dout_155=dout_120^dout_135;
   dout_157=dout_155&dout_113;
   dout_158=dout_155^dout_113;
   dout_159=dout_153|dout_157;
   dout_160=dout_125^dout_136;
   dout_161=dout_125&dout_136;
   dout_162=dout_160&dout_121;
   dout_163=dout_160^dout_121;
   dout_164=dout_161|dout_162;
   dout_165=dout_130^dout_137;
   dout_166=dout_130&dout_137;
   dout_167=dout_165&dout_126;
   dout_168=dout_165^dout_126;
   dout_169=dout_166|dout_167;
   dout_170=dout_96^dout_138;
   dout_171=dout_96&dout_138;
   dout_172=dout_170&dout_131;
   dout_173=dout_170^dout_131;
   dout_174=dout_171|dout_172;
   dout_175=((B >> 0)&1)&((A >> 5)&1);
   dout_176=((B >> 1)&1)&((A >> 5)&1);
   dout_177=((B >> 2)&1)&((A >> 5)&1);
   dout_178=((B >> 3)&1)&((A >> 5)&1);
   dout_179=((B >> 4)&1)&((A >> 5)&1);
   dout_180=((B >> 5)&1)&((A >> 5)&1);
   dout_181=((B >> 6)&1)&((A >> 5)&1);
   dout_182=((B >> 7)&1)&((A >> 5)&1);
   dout_183=dout_145^dout_175;
   dout_184=dout_145&dout_175;
   dout_188=dout_150^dout_176;
   dout_189=dout_150&dout_176;
   dout_190=dout_188&dout_146;
   dout_191=dout_188^dout_146;
   dout_192=dout_189|dout_190;
   dout_193=dout_158^dout_177;
   dout_194=dout_158&dout_177;
   dout_195=dout_193&dout_151;
   dout_196=dout_193^dout_151;
   dout_197=dout_194|dout_195;
   dout_198=dout_163^dout_178;
   dout_199=dout_163&dout_178;
   dout_200=dout_198&dout_159;
   dout_201=dout_198^dout_159;
   dout_202=dout_199|dout_200;
   dout_203=dout_168^dout_179;
   dout_204=dout_168&dout_179;
   dout_205=dout_203&dout_164;
   dout_206=dout_203^dout_164;
   dout_207=dout_204|dout_205;
   dout_208=dout_173^dout_180;
   dout_209=dout_173&dout_180;
   dout_210=dout_208&dout_169;
   dout_211=dout_208^dout_169;
   dout_212=dout_209|dout_210;
   dout_213=dout_139^dout_181;
   dout_214=dout_139&dout_181;
   dout_215=dout_213&dout_174;
   dout_216=dout_213^dout_174;
   dout_217=dout_214|dout_215;
   dout_218=((B >> 0)&1)&((A >> 6)&1);
   dout_219=((B >> 1)&1)&((A >> 6)&1);
   dout_220=((B >> 2)&1)&((A >> 6)&1);
   dout_221=((B >> 3)&1)&((A >> 6)&1);
   dout_222=((B >> 4)&1)&((A >> 6)&1);
   dout_223=((B >> 5)&1)&((A >> 6)&1);
   dout_224=((B >> 6)&1)&((A >> 6)&1);
   dout_225=((B >> 7)&1)&((A >> 6)&1);
   dout_226=dout_191^dout_218;
   dout_227=dout_191&dout_218;
   dout_228=dout_226&dout_184;
   dout_229=dout_226^dout_184;
   dout_230=dout_227|dout_228;
   dout_231=dout_196^dout_219;
   dout_232=dout_196&dout_219;
   dout_233=dout_231&dout_192;
   dout_234=dout_231^dout_192;
   dout_235=dout_232|dout_233;
   dout_236=dout_201^dout_220;
   dout_237=dout_201&dout_220;
   dout_238=dout_236&dout_197;
   dout_239=dout_236^dout_197;
   dout_240=dout_237|dout_238;
   dout_241=dout_206^dout_221;
   dout_242=dout_206&dout_221;
   dout_243=dout_241&dout_202;
   dout_244=dout_241^dout_202;
   dout_245=dout_242|dout_243;
   dout_246=dout_211^dout_222;
   dout_247=dout_211&dout_222;
   dout_248=dout_246&dout_207;
   dout_249=dout_246^dout_207;
   dout_250=dout_247|dout_248;
   dout_251=dout_216^dout_223;
   dout_252=dout_216&dout_223;
   dout_253=dout_251&dout_212;
   dout_254=dout_251^dout_212;
   dout_255=dout_252|dout_253;
   dout_256=dout_182^dout_224;
   dout_257=dout_182&dout_224;
   dout_258=dout_256&dout_217;
   dout_259=dout_256^dout_217;
   dout_260=dout_257|dout_258;
   dout_261=((B >> 0)&1)&((A >> 7)&1);
   dout_262=((B >> 1)&1)&((A >> 7)&1);
   dout_263=((B >> 2)&1)&((A >> 7)&1);
   dout_264=((B >> 3)&1)&((A >> 7)&1);
   dout_265=((B >> 4)&1)&((A >> 7)&1);
   dout_266=((B >> 5)&1)&((A >> 7)&1);
   dout_267=((B >> 6)&1)&((A >> 7)&1);
   dout_268=((B >> 7)&1)&((A >> 7)&1);
   dout_269=dout_234^dout_261;
   dout_270=dout_234&dout_261;
   dout_271=dout_269&dout_230;
   dout_272=dout_269^dout_230;
   dout_273=dout_270|dout_271;
   dout_274=dout_239^dout_262;
   dout_275=dout_239&dout_262;
   dout_276=dout_274&dout_235;
   dout_277=dout_274^dout_235;
   dout_278=dout_275|dout_276;
   dout_279=dout_244^dout_263;
   dout_280=dout_244&dout_263;
   dout_281=dout_279&dout_240;
   dout_282=dout_279^dout_240;
   dout_283=dout_280|dout_281;
   dout_284=dout_249^dout_264;
   dout_285=dout_249&dout_264;
   dout_286=dout_284&dout_245;
   dout_287=dout_284^dout_245;
   dout_288=dout_285|dout_286;
   dout_289=dout_254^dout_265;
   dout_290=dout_254&dout_265;
   dout_291=dout_289&dout_250;
   dout_292=dout_289^dout_250;
   dout_293=dout_290|dout_291;
   dout_294=dout_259^dout_266;
   dout_295=dout_259&dout_266;
   dout_296=dout_294&dout_255;
   dout_297=dout_294^dout_255;
   dout_298=dout_295|dout_296;
   dout_299=dout_225^dout_267;
   dout_300=dout_225&dout_267;
   dout_301=dout_299&dout_260;
   dout_302=dout_299^dout_260;
   dout_303=dout_300|dout_301;
   dout_304=dout_277^dout_273;
   dout_305=dout_277&dout_273;
   dout_306=dout_282^dout_278;
   dout_307=dout_282&dout_278;
   dout_308=dout_306&dout_305;
   dout_309=dout_306^dout_305;
   dout_310=dout_307|dout_308;
   dout_311=dout_287^dout_283;
   dout_312=dout_287&dout_283;
   dout_313=dout_311&dout_310;
   dout_314=dout_311^dout_310;
   dout_315=dout_312|dout_313;
   dout_316=dout_292^dout_288;
   dout_317=dout_292&dout_288;
   dout_318=dout_316&dout_315;
   dout_319=dout_316^dout_315;
   dout_320=dout_317|dout_318;
   dout_321=dout_297^dout_293;
   dout_322=dout_297&dout_293;
   dout_323=dout_321&dout_320;
   dout_324=dout_321^dout_320;
   dout_325=dout_322|dout_323;
   dout_326=dout_302^dout_298;
   dout_327=dout_302&dout_298;
   dout_328=dout_326&dout_325;
   dout_329=dout_326^dout_325;
   dout_330=dout_327|dout_328;
   dout_331=dout_268^dout_303;
   dout_332=((A >> 7)&1)&dout_303;
   dout_333=dout_331&dout_330;
   dout_334=dout_331^dout_330;
   dout_335=dout_332|dout_333;

   O = 0;
   O |= (dout_81&1) << 0;
   O |= (0&1) << 1;
   O |= (dout_50&1) << 2;
   O |= (dout_89&1) << 3;
   O |= (dout_140&1) << 4;
   O |= (dout_183&1) << 5;
   O |= (dout_229&1) << 6;
   O |= (dout_272&1) << 7;
   O |= (dout_304&1) << 8;
   O |= (dout_309&1) << 9;
   O |= (dout_314&1) << 10;
   O |= (dout_319&1) << 11;
   O |= (dout_324&1) << 12;
   O |= (dout_329&1) << 13;
   O |= (dout_334&1) << 14;
   O |= (dout_335&1) << 15;
   return O;
}

__device__ uint64_t mul8u_18UH(const uint64_t B,const uint64_t A)
{
   uint64_t O, dout_73, dout_118, dout_162, dout_163, dout_191, dout_192, dout_207, dout_208, dout_236, dout_237, dout_238, dout_239, dout_240, dout_241, dout_243, dout_244, dout_250, dout_251, dout_252, dout_253, dout_276, dout_277, dout_278, dout_279, dout_280, dout_281, dout_282, dout_283, dout_284, dout_285, dout_286, dout_287, dout_288, dout_289, dout_290, dout_294, dout_295, dout_296, dout_297, dout_298, dout_316, dout_317, dout_318, dout_319, dout_320, dout_321, dout_322, dout_323, dout_324, dout_325, dout_326, dout_327, dout_328, dout_329, dout_330, dout_331, dout_332, dout_333, dout_334, dout_335;   int avg=0;

   dout_73=((A >> 5)&1)&((B >> 5)&1);
   dout_118=((B >> 7)&1)&((A >> 3)&1);
   dout_162=((B >> 6)&1)&((A >> 4)&1);
   dout_163=((B >> 7)&1)&((A >> 4)&1);
   dout_191=dout_118^dout_162;
   dout_192=dout_118&dout_162;
   dout_207=((B >> 6)&1)&((A >> 5)&1);
   dout_208=((B >> 7)&1)&((A >> 5)&1);
   dout_236=dout_163^dout_207;
   dout_237=dout_163&dout_207;
   dout_238=dout_236&dout_191;
   dout_239=dout_236^dout_191;
   dout_240=dout_237|dout_238;
   dout_241=dout_192^dout_208;
   dout_243=dout_208&dout_240;
   dout_244=dout_241^dout_240;
   dout_250=((B >> 4)&1)&((A >> 6)&1);
   dout_251=((B >> 5)&1)&((A >> 6)&1);
   dout_252=((B >> 6)&1)&((A >> 6)&1);
   dout_253=((B >> 7)&1)&((A >> 6)&1);
   dout_276=dout_239^dout_251;
   dout_277=dout_239&dout_251;
   dout_278=dout_276&dout_250;
   dout_279=dout_276^dout_250;
   dout_280=dout_277|dout_278;
   dout_281=dout_244^dout_252;
   dout_282=dout_244&dout_252;
   dout_283=dout_281&dout_280;
   dout_284=dout_281^dout_280;
   dout_285=dout_282|dout_283;
   dout_286=dout_243^dout_253;
   dout_287=dout_243&dout_253;
   dout_288=dout_253&dout_285;
   dout_289=dout_286^dout_285;
   dout_290=dout_287|dout_288;
   dout_294=((B >> 3)&1)&((A >> 7)&1);
   dout_295=((B >> 4)&1)&((A >> 7)&1);
   dout_296=((B >> 5)&1)&((A >> 7)&1);
   dout_297=((B >> 6)&1)&((A >> 7)&1);
   dout_298=((B >> 7)&1)&((A >> 7)&1);
   dout_316=dout_279^dout_295;
   dout_317=dout_279&dout_295;
   dout_318=dout_316&dout_73;
   dout_319=dout_316^dout_73;
   dout_320=dout_317|dout_318;
   dout_321=dout_284^dout_296;
   dout_322=dout_284&dout_296;
   dout_323=dout_321&dout_320;
   dout_324=dout_321^dout_320;
   dout_325=dout_322|dout_323;
   dout_326=dout_289^dout_297;
   dout_327=dout_289&dout_297;
   dout_328=dout_326&dout_325;
   dout_329=dout_326^dout_325;
   dout_330=dout_327|dout_328;
   dout_331=dout_290^dout_298;
   dout_332=dout_290&((A >> 7)&1);
   dout_333=dout_298&dout_330;
   dout_334=dout_331^dout_330;
   dout_335=dout_332|dout_333;

   O = 0;
   O |= (0&1) << 0;
   O |= (dout_208&1) << 1;
   O |= (dout_329&1) << 2;
   O |= (0&1) << 3;
   O |= (dout_208&1) << 4;
   O |= (dout_294&1) << 5;
   O |= (0&1) << 6;
   O |= (dout_251&1) << 7;
   O |= (dout_294&1) << 8;
   O |= (0&1) << 9;
   O |= (dout_294&1) << 10;
   O |= (dout_319&1) << 11;
   O |= (dout_324&1) << 12;
   O |= (dout_329&1) << 13;
   O |= (dout_334&1) << 14;
   O |= (dout_335&1) << 15;
   return O;
}

__device__ sfp_mul custom_multiplier_GTR(sfp_mul a, sfp_mul b)
{
    sfp_mul result;
    result.f = 0;

    int32_t a_exp_bits;
    int32_t b_exp_bits;
    uint64_t a_mant_bits;
    uint64_t b_mant_bits;

    result.p.sign_bit = a.p.sign_bit ^ b.p.sign_bit;

    // exception detection
    if ((a.p.exp_bits == 0 && a.p.mant_bits == 0) || (b.p.exp_bits == 0 && b.p.mant_bits == 0))
    {
        return result;
    }

    int BITMASK_EXPLEN = BITMASK(MUL_EXPLEN);
    int BITMASK_MANTLEN = BITMASK(MUL_MANTLEN);

    // Unpacking
    a_exp_bits = (a.p.exp_bits) ? a.p.exp_bits : a.p.exp_bits - (MUL_MANTLEN - GETLOP(a.p.mant_bits) - 1);
    b_exp_bits = (b.p.exp_bits) ? b.p.exp_bits : b.p.exp_bits - (MUL_MANTLEN - GETLOP(b.p.mant_bits) - 1);
    a_mant_bits = (a.p.exp_bits) ? a.p.mant_bits | ((int32_t)1 << MUL_MANTLEN) : a.p.mant_bits << (MUL_MANTLEN - GETLOP(a.p.mant_bits));
    b_mant_bits = (b.p.exp_bits) ? b.p.mant_bits | ((int32_t)1 << MUL_MANTLEN) : b.p.mant_bits << (MUL_MANTLEN - GETLOP(b.p.mant_bits));

    // 1. calculate the exponent of the product
    int32_t c_exp = a_exp_bits + b_exp_bits - BITMASK(MUL_EXPLEN - 1);

    // 2. multiplication
    //uint64_t m_mant = a_mant_bits * b_mant_bits;

    long long m_mant;
    m_mant = mul8u_GTR(a_mant_bits, b_mant_bits);

    int32_t n_mant = m_mant >> (MUL_MANTLEN + 1);

    // normalize 1
    int shift_amt = MUL_MANTLEN - GETLOP(n_mant);
    if (n_mant != 0 && !GETBIT(n_mant, MUL_MANTLEN))
    {
        c_exp = c_exp - shift_amt;
        n_mant = n_mant << shift_amt;
    }

    // normalize 2
    if (c_exp < 0)
    {
        shift_amt = -c_exp;
        n_mant = (shift_amt < 24) ? n_mant >> shift_amt : 0;
        c_exp = 0;
    }

    // 4. round
    int32_t r_mant = n_mant;

    // 5. if need round up
    if (r_mant > BITMASK_MANTLEN)
        c_exp++;
    if (r_mant > BITMASK(MUL_MANTLEN + 1))
        c_exp++;

    // 5. determine the sign and packing
    if (c_exp >= BITMASK_EXPLEN)
    {
        result.p.exp_bits = BITMASK_EXPLEN;
        result.p.mant_bits = 0;
        return result;
    }

    result.p.exp_bits = c_exp;
    result.p.mant_bits = (result.p.exp_bits >= BITMASK_EXPLEN) ? 0 : r_mant & BITMASK_MANTLEN;
    return result;
}

__device__ sfp_mul custom_multiplier_18UH(sfp_mul a, sfp_mul b)
{
    sfp_mul result;
    result.f = 0;

    int32_t a_exp_bits;
    int32_t b_exp_bits;
    uint64_t a_mant_bits;
    uint64_t b_mant_bits;

    result.p.sign_bit = a.p.sign_bit ^ b.p.sign_bit;

    // exception detection
    if ((a.p.exp_bits == 0 && a.p.mant_bits == 0) || (b.p.exp_bits == 0 && b.p.mant_bits == 0))
    {
        return result;
    }

    int BITMASK_EXPLEN = BITMASK(MUL_EXPLEN);
    int BITMASK_MANTLEN = BITMASK(MUL_MANTLEN);

    // Unpacking
    a_exp_bits = (a.p.exp_bits) ? a.p.exp_bits : a.p.exp_bits - (MUL_MANTLEN - GETLOP(a.p.mant_bits) - 1);
    b_exp_bits = (b.p.exp_bits) ? b.p.exp_bits : b.p.exp_bits - (MUL_MANTLEN - GETLOP(b.p.mant_bits) - 1);
    a_mant_bits = (a.p.exp_bits) ? a.p.mant_bits | ((int32_t)1 << MUL_MANTLEN) : a.p.mant_bits << (MUL_MANTLEN - GETLOP(a.p.mant_bits));
    b_mant_bits = (b.p.exp_bits) ? b.p.mant_bits | ((int32_t)1 << MUL_MANTLEN) : b.p.mant_bits << (MUL_MANTLEN - GETLOP(b.p.mant_bits));

    // 1. calculate the exponent of the product
    int32_t c_exp = a_exp_bits + b_exp_bits - BITMASK(MUL_EXPLEN - 1);

    // 2. multiplication
    //uint64_t m_mant = a_mant_bits * b_mant_bits;

    long long m_mant;
    m_mant = mul8u_18UH(a_mant_bits, b_mant_bits);

    int32_t n_mant = m_mant >> (MUL_MANTLEN + 1);

    // normalize 1
    int shift_amt = MUL_MANTLEN - GETLOP(n_mant);
    if (n_mant != 0 && !GETBIT(n_mant, MUL_MANTLEN))
    {
        c_exp = c_exp - shift_amt;
        n_mant = n_mant << shift_amt;
    }

    // normalize 2
    if (c_exp < 0)
    {
        shift_amt = -c_exp;
        n_mant = (shift_amt < 24) ? n_mant >> shift_amt : 0;
        c_exp = 0;
    }

    // 4. round
    int32_t r_mant = n_mant;

    // 5. if need round up
    if (r_mant > BITMASK_MANTLEN)
        c_exp++;
    if (r_mant > BITMASK(MUL_MANTLEN + 1))
        c_exp++;

    // 5. determine the sign and packing
    if (c_exp >= BITMASK_EXPLEN)
    {
        result.p.exp_bits = BITMASK_EXPLEN;
        result.p.mant_bits = 0;
        return result;
    }

    result.p.exp_bits = c_exp;
    result.p.mant_bits = (result.p.exp_bits >= BITMASK_EXPLEN) ? 0 : r_mant & BITMASK_MANTLEN;
    return result;
}

__device__ sfp_add custom_adder(sfp_add a, sfp_add b)
{
    sfp_add s;

    int32_t a_mant_bits;
    int32_t a_exp_bits;
    int32_t a_sign;

    int32_t b_mant_bits;
    int32_t b_exp_bits;
    int32_t b_sign;

    int32_t s_mant_bits;

    int32_t g_exp;       // Exponent Gap (Difference)
    int32_t n_shift_amt; // Shift Amount for Normalization

    int32_t grs_bits;

    //  0. Preparation - Unpacking
    a_sign = a.p.sign_bit;
    b_sign = b.p.sign_bit;
    a_exp_bits = (a.p.exp_bits) ? a.p.exp_bits : (a.p.mant_bits) ? a.p.exp_bits - (ADD_MANTLEN - GETLOP(a.p.mant_bits) - 1) : 0;
    a_mant_bits = (a.p.exp_bits) ? a.p.mant_bits | ((int32_t)1 << ADD_MANTLEN) : (a.p.mant_bits) ? a.p.mant_bits << (ADD_MANTLEN - GETLOP(a.p.mant_bits)) : 0;
    b_exp_bits = (b.p.exp_bits) ? b.p.exp_bits : (b.p.mant_bits) ? b.p.exp_bits - (ADD_MANTLEN - GETLOP(b.p.mant_bits) - 1) : 0;
    b_mant_bits = (b.p.exp_bits) ? b.p.mant_bits | ((int32_t)1 << ADD_MANTLEN) : (b.p.mant_bits) ? b.p.mant_bits << (ADD_MANTLEN - GETLOP(b.p.mant_bits)) : 0;

    // 1. Exponent Difference (Gap)
    g_exp = a_exp_bits + (~b_exp_bits + 1); // a_exp_bits - b_exp_bits;

    // 2. Selective Complement and Possible SWAP to make a is always greater than b
    a_mant_bits = (a_sign && !b_sign) ? ~a_mant_bits + 1 : a_mant_bits;
    b_mant_bits = (!a_sign && b_sign) ? ~b_mant_bits + 1 : b_mant_bits;

    if (g_exp < 0)
    {
        SWAP(a_mant_bits, b_mant_bits, int32_t);
        SWAP(a_exp_bits, b_exp_bits, int32_t);
        SWAP(a_sign, b_sign, int32_t);
        g_exp = ~g_exp + 1; // g_exp = -g_exp;
    }

    // 3. Align Mantissa
    a_mant_bits <<= GRSLEN; // reserve lower 3 bits for GRS
    b_mant_bits <<= GRSLEN; // reserve lower 3 bits for GRS

    grs_bits = b_mant_bits & (BITMASK(g_exp - 2) << GRSLEN);                  // for Sticky Bit  -2 for GR bit
    b_mant_bits = (g_exp >= ADD_MANTLEN + GRSLEN) ? 0 : b_mant_bits >> g_exp; // No possibility to round if the MSB is shifted beyond G position
    b_mant_bits = (grs_bits) ? b_mant_bits | 0x1 : b_mant_bits;               // Add Sticky

    //  4. Mantissa Addition (24+3 bit addition) and Sign/Complement
    s_mant_bits = a_mant_bits + b_mant_bits; // MANTLEN+1 + 3 bits
    s.p.sign_bit = (s_mant_bits < 0 || (a_sign && b_sign)) ? 1 : 0;
    s_mant_bits = (s_mant_bits < 0) ? ~s_mant_bits + 1 : s_mant_bits;

    // 5. Normalization (Leading One Position Detection & Shift)
    n_shift_amt = (s_mant_bits) ? GETLOP(s_mant_bits) - (ADD_MANTLEN + GRSLEN) : -a_exp_bits; // Leading 1 Position Detection
    if (n_shift_amt > 0)                                                                      // n_shift_amt == 1
    {
        grs_bits = (s_mant_bits >> 1) & BITMASK(GRSLEN) | GETBIT(s_mant_bits, 0); // G=3rd, R=2nd, S=1st | 0th bit position
        s_mant_bits >>= 1;
    }
    else
    {
        s_mant_bits <<= -n_shift_amt;
        grs_bits = s_mant_bits & BITMASK(GRSLEN); // G=2nd, R=1st, S=0th bit position
    }

    // 6. Rounding
    if (grs_bits == 0x4)                                                // GRS == 'b100
        s_mant_bits = (s_mant_bits >> GRSLEN) + GETBIT(s_mant_bits, 3); // Round to nearst even
    else
        s_mant_bits = (s_mant_bits >> GRSLEN) + GETBIT(s_mant_bits, 2); // Add Guard bit

    // 7. Normalization if needed
    if (GETBIT(s_mant_bits, ADD_MANTLEN + 1))
    {
        s_mant_bits >>= 1;
        n_shift_amt++;
    }

    // 8. Packing
    s.p.exp_bits = (a_exp_bits + n_shift_amt <= 0) ? 0 : (a_exp_bits + n_shift_amt >= BITMASK(ADD_EXPLEN)) ? BITMASK(ADD_EXPLEN)
                                                                                                           : a_exp_bits + n_shift_amt;
    s.p.mant_bits = (a_exp_bits + n_shift_amt <= 0) ? s_mant_bits >> (1 - a_exp_bits - n_shift_amt) : (s.p.exp_bits == BITMASK(ADD_EXPLEN)) ? 0
                                                                                                                                            : s_mant_bits & BITMASK(ADD_MANTLEN);

    return s;
}

__device__ sfp_add loa_fpadder(sfp_add a, sfp_add b)
{
    sfp_add s;

    int32_t a_mant_bits;
    int32_t a_exp_bits;
    int32_t a_sign;

    int32_t b_mant_bits;
    int32_t b_exp_bits;
    int32_t b_sign;

    int32_t s_mant_bits;

    int32_t g_exp;       // Exponent Gap (Difference)
    int32_t n_shift_amt; // Shift Amount for Normalization

    int32_t grs_bits;

    //  0. Preparation - Unpacking
    a_sign = a.p.sign_bit;
    b_sign = b.p.sign_bit;
    a_exp_bits = (a.p.exp_bits) ? a.p.exp_bits : (a.p.mant_bits) ? a.p.exp_bits - (ADD_MANTLEN - GETLOP(a.p.mant_bits) - 1) : 0;
    a_mant_bits = (a.p.exp_bits) ? a.p.mant_bits | ((int32_t)1 << ADD_MANTLEN) : (a.p.mant_bits) ? a.p.mant_bits << (ADD_MANTLEN - GETLOP(a.p.mant_bits)) : 0;
    b_exp_bits = (b.p.exp_bits) ? b.p.exp_bits : (b.p.mant_bits) ? b.p.exp_bits - (ADD_MANTLEN - GETLOP(b.p.mant_bits) - 1) : 0;
    b_mant_bits = (b.p.exp_bits) ? b.p.mant_bits | ((int32_t)1 << ADD_MANTLEN) : (b.p.mant_bits) ? b.p.mant_bits << (ADD_MANTLEN - GETLOP(b.p.mant_bits)) : 0;

    // 1. Exponent Difference (Gap)
    g_exp = a_exp_bits + (~b_exp_bits + 1); // a_exp_bits - b_exp_bits;

    // 2. Selective Complement and Possible SWAP to make a is always greater than b
    a_mant_bits = (a_sign && !b_sign) ? ~a_mant_bits + 1 : a_mant_bits;
    b_mant_bits = (!a_sign && b_sign) ? ~b_mant_bits + 1 : b_mant_bits;

    if (g_exp < 0)
    {
        SWAP(a_mant_bits, b_mant_bits, int32_t);
        SWAP(a_exp_bits, b_exp_bits, int32_t);
        SWAP(a_sign, b_sign, int32_t);
        g_exp = ~g_exp + 1; // g_exp = -g_exp;
    }

    b_mant_bits = (g_exp >= ADD_MANTLEN) ? 0 : b_mant_bits >> g_exp; // No possibility to round if the MSB is shifted beyond G position

    // 3. LOA
    int32_t m_a, n_a; //m is exact part, n is approximate part
    int32_t m_b, n_b; //m is exact part, n is approximate part

    m_a = a_mant_bits >> LOA_SIZE;
    m_b = b_mant_bits >> LOA_SIZE;
    n_a = a_mant_bits & BITMASK(LOA_SIZE);
    n_b = b_mant_bits & BITMASK(LOA_SIZE);

    s_mant_bits = m_a + m_b;
    int32_t c = GETBIT(n_a, LOA_SIZE-1) & GETBIT(n_b, LOA_SIZE-1); //carry
    s_mant_bits = (s_mant_bits + c) << LOA_SIZE;
    s_mant_bits = s_mant_bits | (n_a | n_b);
    s.p.sign_bit = (s_mant_bits < 0 || (a_sign && b_sign)) ? 1 : 0;
    s_mant_bits = (s_mant_bits < 0) ? ~s_mant_bits + 1 : s_mant_bits;

    // // 5. Normalization (Leading One Position Detection & Shift)
    n_shift_amt = (s_mant_bits) ? GETLOP(s_mant_bits) - (ADD_MANTLEN) : -a_exp_bits; // Leading 1 Position Detection
    if (n_shift_amt > 0)
    {
        s_mant_bits >>= 1;
    }
    else
    {
        s_mant_bits <<= -n_shift_amt;
    }

    // 7. Normalization if needed
    if (GETBIT(s_mant_bits, ADD_MANTLEN + 1))
    {
        s_mant_bits >>= 1;
        n_shift_amt++;
    }

    // 8. Packing
    s.p.exp_bits = (a_exp_bits + n_shift_amt <= 0) ? 0 : (a_exp_bits + n_shift_amt >= BITMASK(ADD_EXPLEN)) ? BITMASK(ADD_EXPLEN) : a_exp_bits + n_shift_amt;
    s.p.mant_bits = (a_exp_bits + n_shift_amt <= 0) ? s_mant_bits >> (1 - a_exp_bits - n_shift_amt) : (s.p.exp_bits == BITMASK(ADD_EXPLEN)) ? 0 : s_mant_bits & BITMASK(ADD_MANTLEN);

    return s;
}

__device__ sfp_add ama5_fpadder(sfp_add a, sfp_add b)
{
    sfp_add s;

    int32_t a_mant_bits;
    int32_t a_exp_bits;
    int32_t a_sign;

    int32_t b_mant_bits;
    int32_t b_exp_bits;
    int32_t b_sign;

    int32_t s_mant_bits;

    int32_t g_exp;       // Exponent Gap (Difference)
    int32_t n_shift_amt; // Shift Amount for Normalization

    int32_t grs_bits;

    //  0. Preparation - Unpacking
    a_sign = a.p.sign_bit;
    b_sign = b.p.sign_bit;
    a_exp_bits = (a.p.exp_bits) ? a.p.exp_bits : (a.p.mant_bits) ? a.p.exp_bits - (ADD_MANTLEN - GETLOP(a.p.mant_bits) - 1) : 0;
    a_mant_bits = (a.p.exp_bits) ? a.p.mant_bits | ((int32_t)1 << ADD_MANTLEN) : (a.p.mant_bits) ? a.p.mant_bits << (ADD_MANTLEN - GETLOP(a.p.mant_bits)) : 0;
    b_exp_bits = (b.p.exp_bits) ? b.p.exp_bits : (b.p.mant_bits) ? b.p.exp_bits - (ADD_MANTLEN - GETLOP(b.p.mant_bits) - 1) : 0;
    b_mant_bits = (b.p.exp_bits) ? b.p.mant_bits | ((int32_t)1 << ADD_MANTLEN) : (b.p.mant_bits) ? b.p.mant_bits << (ADD_MANTLEN - GETLOP(b.p.mant_bits)) : 0;

    // 1. Exponent Difference (Gap)
    g_exp = a_exp_bits + (~b_exp_bits + 1); // a_exp_bits - b_exp_bits;

    // 2. Selective Complement and Possible SWAP to make a is always greater than b
    a_mant_bits = (a_sign && !b_sign) ? ~a_mant_bits + 1 : a_mant_bits;
    b_mant_bits = (!a_sign && b_sign) ? ~b_mant_bits + 1 : b_mant_bits;

    if (g_exp < 0)
    {
        SWAP(a_mant_bits, b_mant_bits, int32_t);
        SWAP(a_exp_bits, b_exp_bits, int32_t);
        SWAP(a_sign, b_sign, int32_t);
        g_exp = ~g_exp + 1; // g_exp = -g_exp;
    }

    b_mant_bits = (g_exp >= ADD_MANTLEN) ? 0 : b_mant_bits >> g_exp; // No possibility to round if the MSB is shifted beyond G position

    // 3. AMA5
    int32_t m_a, n_a; //m is exact part, n is approximate part
    int32_t m_b, n_b; //m is exact part, n is approximate part

    m_a = a_mant_bits >> AMA5_SIZE;
    m_b = b_mant_bits >> AMA5_SIZE;
    n_a = a_mant_bits & BITMASK(AMA5_SIZE);
    n_b = b_mant_bits & BITMASK(AMA5_SIZE);
    
    s_mant_bits = m_a + m_b;
    int32_t c = GETBIT(n_a, AMA5_SIZE-1); //carry
    s_mant_bits = (s_mant_bits + c) << AMA5_SIZE;
    s_mant_bits = s_mant_bits | n_b;

    s.p.sign_bit = (s_mant_bits < 0 || (a_sign && b_sign)) ? 1 : 0;
    s_mant_bits = (s_mant_bits < 0) ? ~s_mant_bits + 1 : s_mant_bits;

    // // 5. Normalization (Leading One Position Detection & Shift)
    n_shift_amt = (s_mant_bits) ? GETLOP(s_mant_bits) - (ADD_MANTLEN) : -a_exp_bits; // Leading 1 Position Detection
    if (n_shift_amt > 0)
    {
        s_mant_bits >>= 1;
    }
    else
    {
        s_mant_bits <<= -n_shift_amt;
    }

    // 7. Normalization if needed
    if (GETBIT(s_mant_bits, ADD_MANTLEN + 1))
    {
        s_mant_bits >>= 1;
        n_shift_amt++;
    }

    // 8. Packing
    s.p.exp_bits = (a_exp_bits + n_shift_amt <= 0) ? 0 : (a_exp_bits + n_shift_amt >= BITMASK(ADD_EXPLEN)) ? BITMASK(ADD_EXPLEN) : a_exp_bits + n_shift_amt;
    s.p.mant_bits = (a_exp_bits + n_shift_amt <= 0) ? s_mant_bits >> (1 - a_exp_bits - n_shift_amt) : (s.p.exp_bits == BITMASK(ADD_EXPLEN)) ? 0 : s_mant_bits & BITMASK(ADD_MANTLEN);

    return s;
}

__device__ sfp_add eta1_fpadder(sfp_add a, sfp_add b)
{
        sfp_add s;

    int32_t a_mant_bits;
    int32_t a_exp_bits;
    int32_t a_sign;

    int32_t b_mant_bits;
    int32_t b_exp_bits;
    int32_t b_sign;

    int32_t s_mant_bits;

    int32_t g_exp;       // Exponent Gap (Difference)
    int32_t n_shift_amt; // Shift Amount for Normalization

    int32_t grs_bits;

    //  0. Preparation - Unpacking
    a_sign = a.p.sign_bit;
    b_sign = b.p.sign_bit;
    a_exp_bits = (a.p.exp_bits) ? a.p.exp_bits : (a.p.mant_bits) ? a.p.exp_bits - (ADD_MANTLEN - GETLOP(a.p.mant_bits) - 1) : 0;
    a_mant_bits = (a.p.exp_bits) ? a.p.mant_bits | ((int32_t)1 << ADD_MANTLEN) : (a.p.mant_bits) ? a.p.mant_bits << (ADD_MANTLEN - GETLOP(a.p.mant_bits)) : 0;
    b_exp_bits = (b.p.exp_bits) ? b.p.exp_bits : (b.p.mant_bits) ? b.p.exp_bits - (ADD_MANTLEN - GETLOP(b.p.mant_bits) - 1) : 0;
    b_mant_bits = (b.p.exp_bits) ? b.p.mant_bits | ((int32_t)1 << ADD_MANTLEN) : (b.p.mant_bits) ? b.p.mant_bits << (ADD_MANTLEN - GETLOP(b.p.mant_bits)) : 0;

    // 1. Exponent Difference (Gap)
    g_exp = a_exp_bits + (~b_exp_bits + 1); // a_exp_bits - b_exp_bits;

    // 2. Selective Complement and Possible SWAP to make a is always greater than b
    a_mant_bits = (a_sign && !b_sign) ? ~a_mant_bits + 1 : a_mant_bits;
    b_mant_bits = (!a_sign && b_sign) ? ~b_mant_bits + 1 : b_mant_bits;

    if (g_exp < 0)
    {
        SWAP(a_mant_bits, b_mant_bits, int32_t);
        SWAP(a_exp_bits, b_exp_bits, int32_t);
        SWAP(a_sign, b_sign, int32_t);
        g_exp = ~g_exp + 1; // g_exp = -g_exp;
    }

    b_mant_bits = (g_exp >= ADD_MANTLEN) ? 0 : b_mant_bits >> g_exp; // No possibility to round if the MSB is shifted beyond G position

    // 3. ETA1
    int32_t m_a, n_a; //m is exact part, n is approximate part
    int32_t m_b, n_b; //m is exact part, n is approximate part

    m_a = a_mant_bits >> ETA1_SIZE;
    m_b = b_mant_bits >> ETA1_SIZE;
    n_a = a_mant_bits & BITMASK(ETA1_SIZE);
    n_b = b_mant_bits & BITMASK(ETA1_SIZE);

    s_mant_bits = m_a + m_b;
    int bitmask_len = 0;
    for(int i=ETA1_SIZE-1; i>=0; i--)
    {
        if(GETBIT(n_a, i) == 1 && GETBIT(n_b, i) == 1)
        {
            bitmask_len = i+1;
            break;
        }
    }
    n_a = n_a ^ n_b | BITMASK(bitmask_len);
    s_mant_bits = s_mant_bits << ETA1_SIZE;
    s_mant_bits = s_mant_bits | n_a;
    
    s.p.sign_bit = (s_mant_bits < 0 || (a_sign && b_sign)) ? 1 : 0;
    s_mant_bits = (s_mant_bits < 0) ? ~s_mant_bits + 1 : s_mant_bits;

    // // 5. Normalization (Leading One Position Detection & Shift)
    n_shift_amt = (s_mant_bits) ? GETLOP(s_mant_bits) - (ADD_MANTLEN) : -a_exp_bits; // Leading 1 Position Detection
    if (n_shift_amt > 0)
    {
        s_mant_bits >>= 1;
    }
    else
    {
        s_mant_bits <<= -n_shift_amt;
    }

    // 7. Normalization if needed
    if (GETBIT(s_mant_bits, ADD_MANTLEN + 1))
    {
        s_mant_bits >>= 1;
        n_shift_amt++;
    }

    // 8. Packing
    s.p.exp_bits = (a_exp_bits + n_shift_amt <= 0) ? 0 : (a_exp_bits + n_shift_amt >= BITMASK(ADD_EXPLEN)) ? BITMASK(ADD_EXPLEN) : a_exp_bits + n_shift_amt;
    s.p.mant_bits = (a_exp_bits + n_shift_amt <= 0) ? s_mant_bits >> (1 - a_exp_bits - n_shift_amt) : (s.p.exp_bits == BITMASK(ADD_EXPLEN)) ? 0 : s_mant_bits & BITMASK(ADD_MANTLEN);

    return s;
}
