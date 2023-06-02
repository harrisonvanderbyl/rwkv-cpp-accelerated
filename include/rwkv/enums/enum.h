#pragma once
enum MODE {
    PARRALEL,
    GPT
};

enum
{
    X,
    EMBED,
    LAYERNORMS,
    STATEXY,
    STATEAA,
    STATEBB,
    STATEPP,
    STATEDD,
    BUFFER1,
    BUFFER2,
    BUFFER3,
    BUFFER4,
    MIXK,
    MIXV,
    MIXR,
    KM,
    VM,
    RM,
    KR,
    VR,
    RR,
    O1,
    O2,
    O3,
    ATTOUT,
    ATTOUTR,
    ATTOUTO,
    FFNMIXK,
    FFNMIXV,
    FFNK,
    FFNV,
    FFNR,
    FFNKR,
    FFNVR,
    FFNRR,
    FFNKO,
    FFNVO,
    FFNRO,
    FFNKBUFFER,
    FFNVBUFFER,
    FFNRBUFFER,
    DECAY,
    BONUS,
    HEAD,
    HEADR,
    HEADO
};

#define VOCAB 65536