#include "gadgetconfig.h"
// Native global snapshot diagnostic; no raw padding or runtime handles.
#ifdef RESEARCH_RESTART_STATE_CAPSULE
  write_string_attribute(handle,"ResearchGlobalLayoutSHA256","39ec0e05d9581469d53afe39a8038756b955e46abd342a39ee7948fe02dc6d2a");
#define RESEARCH_GLOBAL_FLOAT(name,value) { const double stored=static_cast<double>(value); write_scalar_attribute(handle,"ResearchGlobal_" name,&stored,H5T_NATIVE_DOUBLE); }
#define RESEARCH_GLOBAL_INTEGER(name,value) { const uint64_t stored=static_cast<uint64_t>(value); write_scalar_attribute(handle,"ResearchGlobal_" name,&stored,H5T_NATIVE_UINT64); }
#define RESEARCH_GLOBAL_TEXT(name,value) write_string_attribute(handle,"ResearchGlobal_" name,value);
#include "research_restart_global_fields.inc"
#undef RESEARCH_GLOBAL_FLOAT
#undef RESEARCH_GLOBAL_INTEGER
#undef RESEARCH_GLOBAL_TEXT
  static const char active_global_fields[]=
#define RESEARCH_GLOBAL_FLOAT(name,value) name ";"
#define RESEARCH_GLOBAL_INTEGER(name,value) name ";"
#define RESEARCH_GLOBAL_TEXT(name,value) name ";"
#include "research_restart_global_fields.inc"
#undef RESEARCH_GLOBAL_FLOAT
#undef RESEARCH_GLOBAL_INTEGER
#undef RESEARCH_GLOBAL_TEXT
    "";
  write_string_attribute(handle,"ResearchGlobalActiveFields",active_global_fields);
  if(All.OutputListLength<0 || All.OutputListLength>MAXLEN_OUTPUTLIST)Terminate("Global diagnostic output list length outside native bounds");
  {
    const int length=TIMEBINS;
    write_scalar_attribute(handle,"ResearchGlobal_Ti_begstep_Length",&length,H5T_NATIVE_INT);
    if(length>0) {
      std::vector<uint64_t> stored(length);
      for(int index=0;index<length;index++)stored[index]=static_cast<uint64_t>(All.Ti_begstep[index]);
      write_vector_attribute(handle,"ResearchGlobal_Ti_begstep",stored.data(),H5T_NATIVE_UINT64,length);
    }
  }
  {
    const int length=NTYPES;
    write_scalar_attribute(handle,"ResearchGlobal_SofteningClassOfPartType_Length",&length,H5T_NATIVE_INT);
    if(length>0) {
      std::vector<uint64_t> stored(length);
      for(int index=0;index<length;index++)stored[index]=static_cast<uint64_t>(All.SofteningClassOfPartType[index]);
      write_vector_attribute(handle,"ResearchGlobal_SofteningClassOfPartType",stored.data(),H5T_NATIVE_UINT64,length);
    }
  }
  {
    const int length=NSOFTCLASSES;
    write_scalar_attribute(handle,"ResearchGlobal_SofteningComoving_Length",&length,H5T_NATIVE_INT);
    if(length>0) {
      write_vector_attribute(handle,"ResearchGlobal_SofteningComoving",All.SofteningComoving,H5T_NATIVE_DOUBLE,length);
    }
  }
  {
    const int length=NSOFTCLASSES;
    write_scalar_attribute(handle,"ResearchGlobal_SofteningMaxPhys_Length",&length,H5T_NATIVE_INT);
    if(length>0) {
      write_vector_attribute(handle,"ResearchGlobal_SofteningMaxPhys",All.SofteningMaxPhys,H5T_NATIVE_DOUBLE,length);
    }
  }
  {
    const int length=NSOFTCLASSES+NSOFTCLASSES_HYDRO;
    write_scalar_attribute(handle,"ResearchGlobal_SofteningTable_Length",&length,H5T_NATIVE_INT);
    if(length>0) {
      write_vector_attribute(handle,"ResearchGlobal_SofteningTable",All.SofteningTable,H5T_NATIVE_DOUBLE,length);
    }
  }
  {
    const int length=NSOFTCLASSES+NSOFTCLASSES_HYDRO+2;
    write_scalar_attribute(handle,"ResearchGlobal_ForceSoftening_Length",&length,H5T_NATIVE_INT);
    if(length>0) {
      write_vector_attribute(handle,"ResearchGlobal_ForceSoftening",All.ForceSoftening,H5T_NATIVE_DOUBLE,length);
    }
  }
  {
    const int length=NTYPES;
    write_scalar_attribute(handle,"ResearchGlobal_MassTable_Length",&length,H5T_NATIVE_INT);
    if(length>0) {
      write_vector_attribute(handle,"ResearchGlobal_MassTable",All.MassTable,H5T_NATIVE_DOUBLE,length);
    }
  }
  {
    const int length=All.OutputListLength;
    write_scalar_attribute(handle,"ResearchGlobal_OutputListTimes_Length",&length,H5T_NATIVE_INT);
    if(length>0) {
      write_vector_attribute(handle,"ResearchGlobal_OutputListTimes",All.OutputListTimes,H5T_NATIVE_DOUBLE,length);
    }
  }
  {
    const int length=All.OutputListLength;
    write_scalar_attribute(handle,"ResearchGlobal_OutputListFlag_Length",&length,H5T_NATIVE_INT);
    if(length>0) {
      std::vector<uint64_t> stored(length);
      for(int index=0;index<length;index++)stored[index]=static_cast<uint64_t>(All.OutputListFlag[index]);
      write_vector_attribute(handle,"ResearchGlobal_OutputListFlag",stored.data(),H5T_NATIVE_UINT64,length);
    }
  }
#endif
