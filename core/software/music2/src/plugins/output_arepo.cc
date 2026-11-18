/*
 * output_arepo.cc - This file is part of MUSIC -
 * a code to generate multi-scale initial conditions 
 * for cosmological simulations
 * 
 * Copyright (C) 2010  Oliver Hahn
 * 
 * Plugin: Dylan Nelson (dnelson@cfa.harvard.edu)
 */

#ifdef HAVE_HDF5

#define GAS_PARTTYPE 0
#define HIGHRES_DM_PARTTYPE 1
#define COARSE_DM_DEFAULT_PARTTYPE 2
#define STAR_PARTTYPE 4
#define NTYPES 6

#include <sstream>
#include <string>
#include <algorithm>
#include <limits>
#include "output.hh"
#include "HDF_IO.hh"
#include "particle_count_header.hh"

class arepo_output_plugin : public output_plugin
{
protected:
  // header/config
  std::vector<std::vector<unsigned long long>> nPart;
  std::vector<long long> nPartTotal;
  std::vector<double> massTable;
  double time, redshift, boxSize;
  unsigned int numFiles, coarsePartType;

  double omega0, omega_L, hubbleParam;

  // configuration
  double UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s;
  double omega_b, rhoCrit;
  double posFac, velFac;
  long long nPartTotAllTypes;
  bool doBaryons, useLongIDs, doublePrec;
  bool shiftBack;
  std::vector<double> coordinateShift;
  std::vector<double> maximumDisplacement;
  std::vector<double> compensationMean = std::vector<double>(6,0.0);
  std::vector<double> meanSpeciesMass = std::vector<double>(6,0.0);
  std::vector<double> removedBulkVelocity = std::vector<double>(3, 0.0);

  size_t npfine, npart, npcoarse;
  std::vector<size_t> levelcounts;

  // parameter file hints
  int pmgrid, gridboost;
  float softening, Tini;

  using output_plugin::cf_;

  // Research2: retain native layout/values, protect every nonempty particle
  // payload against corrupt chunk reads, including coordinate hyperslab writes.
  hid_t createCheckedDataset(hid_t group, const char *name, hid_t type,
                             hid_t space, int rank, const hsize_t *dimensions)
  {
    hid_t properties = H5Pcreate(H5P_DATASET_CREATE);
    if(properties < 0) throw std::runtime_error("Cannot create native checksum properties");
    if(dimensions[0] > 0)
      {
        hsize_t chunks[2] = {std::min<hsize_t>(dimensions[0], 16384), 3};
        if(H5Pset_chunk(properties, rank, chunks) < 0 || H5Pset_fletcher32(properties) < 0)
          { H5Pclose(properties); throw std::runtime_error("Cannot enable native chunk checksum"); }
      }
    hid_t dataset = H5Dcreate(group, name, type, space, properties);
    if(H5Pclose(properties) < 0 || dataset < 0)
      throw std::runtime_error("Cannot create checksum-protected native dataset");
    return dataset;
  }

  // Nx1 vector (e.g. masses,particleids)
  template <typename T>
  void writeHDF5_a(std::string fieldName, int partTypeNum, const std::vector<T> &data)
  {
    hid_t HDF_FileID, HDF_GroupID, HDF_DatasetID, HDF_DataspaceID, HDF_Type;
    hsize_t HDF_Dims, offset = 0;

    std::stringstream GrpName;
    GrpName << "PartType" << partTypeNum;

    for (unsigned i = 0; i < numFiles; i++)
    {
      std::string filename = fname_;
      HDF_Dims = data.size();

      // modify local filename and write size
      if (numFiles > 1)
      {
        std::stringstream s;
        s << "." << i << ".hdf5";
        filename.replace(filename.find(".hdf5"), 5, s.str());

        HDF_Dims = ceil(data.size() / numFiles);
        if (i == numFiles - 1)
          HDF_Dims = data.size() - offset;
      }

      HDF_FileID = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
      HDF_GroupID = H5Gopen(HDF_FileID, GrpName.str().c_str());

      HDF_Type = GetDataType<T>();
      HDF_DataspaceID = H5Screate_simple(1, &HDF_Dims, NULL);
      HDF_DatasetID = createCheckedDataset(HDF_GroupID, fieldName.c_str(), HDF_Type, HDF_DataspaceID, 1, &HDF_Dims);

      // write and close
      H5Dwrite(HDF_DatasetID, HDF_Type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[offset]);

      H5Dclose(HDF_DatasetID);
      H5Sclose(HDF_DataspaceID);

      H5Gclose(HDF_GroupID);
      H5Fclose(HDF_FileID);

      offset += HDF_Dims;
    }
  }

  // Nx3 vector (e.g. pos,vel), where coord = index of the second dimension (written one at a time)
  template <typename T>
  void writeHDF5_b(std::string fieldName, int coord, int partTypeNum, std::vector<T> &data, bool readFlag = false)
  {
    hid_t HDF_FileID, HDF_GroupID, HDF_DatasetID, HDF_DataspaceID, HDF_Type;
    hsize_t HDF_Dims[2], HDF_DimsMem[2], w_offset = 0;

    std::stringstream GrpName;
    GrpName << "PartType" << partTypeNum;

    for (unsigned i = 0; i < numFiles; i++)
    {
      std::string filename = fname_;
      HDF_Dims[0] = data.size();

      // modify local filename and write size
      if (numFiles > 1)
      {
        std::stringstream s;
        s << "." << i << ".hdf5";
        filename.replace(filename.find(".hdf5"), 5, s.str());

        HDF_Dims[0] = ceil(data.size() / numFiles);
        if (i == numFiles - 1)
          HDF_Dims[0] = data.size() - w_offset;
      }

      HDF_FileID = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
      HDF_GroupID = H5Gopen(HDF_FileID, GrpName.str().c_str());

      HDF_Type = GetDataType<T>();
      HDF_Dims[1] = 3;

      // if dataset does not yet exist, create it (on first coord call)
      if (!(H5Lexists(HDF_GroupID, fieldName.c_str(), H5P_DEFAULT)))
      {
        HDF_DataspaceID = H5Screate_simple(2, HDF_Dims, NULL);
        HDF_DatasetID = createCheckedDataset(HDF_GroupID, fieldName.c_str(), HDF_Type, HDF_DataspaceID, 2, HDF_Dims);

        H5Sclose(HDF_DataspaceID);
        H5Dclose(HDF_DatasetID);
      }

      // make memory space (just indicates the size/shape of data)
      HDF_DimsMem[0] = HDF_Dims[0];
      HDF_DimsMem[1] = 1;
      hid_t HDF_MemoryspaceID = H5Screate_simple(2, HDF_DimsMem, NULL);

      // open hyperslab
      hsize_t count[2] = {1, 1}, stride[2] = {1, 1}, offset[2] = {0, 0};

      offset[1] = coord;      // set where in the second dimension to write
      count[0] = HDF_Dims[0]; // set size in the first dimension (num particles of this type)

      HDF_DatasetID = H5Dopen(HDF_GroupID, fieldName.c_str());
      HDF_DataspaceID = H5Dget_space(HDF_DatasetID);

      H5Sselect_hyperslab(HDF_DataspaceID, H5S_SELECT_SET, offset, stride, count, NULL);

      // write (or read) and close
      if (readFlag)
        H5Dread(HDF_DatasetID, HDF_Type, HDF_MemoryspaceID, HDF_DataspaceID, H5P_DEFAULT,
                &data[w_offset]);
      else
        H5Dwrite(HDF_DatasetID, HDF_Type, HDF_MemoryspaceID, HDF_DataspaceID, H5P_DEFAULT,
                 &data[w_offset]);

      H5Dclose(HDF_DatasetID);
      H5Gclose(HDF_GroupID);
      H5Fclose(HDF_FileID);

      w_offset += HDF_Dims[0];
    }
  }

  // called from finalize()
  void generateAndWriteIDs(void)
  {
    unsigned long long offset = 1; // don't use ID==0
    nPartTotAllTypes = 0;

    for (size_t i = 0; i < nPartTotal.size(); i++)
    {
      if (!nPartTotal[i])
        continue;

      if(nPartTotal[i] < 0 || (unsigned long long)nPartTotal[i] > std::numeric_limits<unsigned long long>::max() - offset)
        throw std::runtime_error("Native particle identity namespace exceeds uint64");
      if(nPartTotal[i]>std::numeric_limits<long long>::max()-nPartTotAllTypes)
        throw std::runtime_error("Native total particle count exceeds signed 64-bit storage");
      nPartTotAllTypes += nPartTotal[i];

      if (!useLongIDs)
      {
        if(offset > (unsigned long long)std::numeric_limits<int>::max() ||
           (unsigned long long)nPartTotal[i] - 1 > (unsigned long long)std::numeric_limits<int>::max() - offset)
          throw std::runtime_error("32-bit native particle identities would truncate; enable long IDs");
        std::vector<int> ids = std::vector<int>(nPartTotal[i]);
        for (int j = 0; j < nPartTotal[i]; j++)
          ids[j] = offset + j;

        writeHDF5_a("ParticleIDs", i, ids);
      }
      else
      {
        std::vector<unsigned long long> ids = std::vector<unsigned long long>(nPartTotal[i]);
        for (long long j = 0; j < nPartTotal[i]; j++)
          ids[j] = offset + j;

        writeHDF5_a("ParticleIDs", i, ids);
      }

      // make IDs of all particle types sequential (unique) = unnecessary, but consistent with gadget output format
      offset += nPartTotal[i];
    }
  }

  void countLeafCells(const grid_hierarchy &gh)
  {
    npfine = 0;
    npart = 0;
    npcoarse = 0;

    npfine = gh.count_leaf_cells(gh.levelmax(), gh.levelmax());
    npart = gh.count_leaf_cells(gh.levelmin(), gh.levelmax());

    if (levelmax_ != levelmin_) // multimass
      npcoarse = gh.count_leaf_cells(gh.levelmin(), gh.levelmax() - 1);
  }

  template <typename T>
  void __write_dm_mass(const grid_hierarchy &gh)
  {
    countLeafCells(gh);

    // fill levelcount for header
    levelcounts = std::vector<size_t>(levelmax_ - levelmin_ + 1);
    for (int ilevel = gh.levelmax(); ilevel >= (int)gh.levelmin(); --ilevel)
      levelcounts[gh.levelmax() - ilevel] = gh.count_leaf_cells(ilevel, ilevel);

    if (levelmax_ > levelmin_ + 1) // morethan2bnd
    {
      // DM particles will have variable masses
      size_t count = 0;

      std::vector<T> data(npcoarse);

      for (int ilevel = gh.levelmax() - 1; ilevel >= (int)gh.levelmin(); --ilevel)
      {
        // baryon particles live only on finest grid, these particles here are total matter particles
        T pmass = omega0 * rhoCrit * pow(boxSize * posFac, 3.0) / pow(2, 3 * ilevel);

        for (unsigned i = 0; i < gh.get_grid(ilevel)->size(0); ++i)
          for (unsigned j = 0; j < gh.get_grid(ilevel)->size(1); ++j)
            for (unsigned k = 0; k < gh.get_grid(ilevel)->size(2); ++k)
              if (gh.is_in_mask(ilevel, i, j, k) && !gh.is_refined(ilevel, i, j, k))
              {
                data[count++] = pmass;
              }
      }

      if (count != npcoarse)
        throw std::runtime_error("Internal consistency error while writing masses");

      writeHDF5_a("Masses", coarsePartType, data); // write DM
    }
    else
    {
      // DM particles will all have the same mass, just write to massTable
      if (levelmax_ != levelmin_) // multimass
        massTable[coarsePartType] = omega0 * rhoCrit * pow(boxSize * posFac, 3.0) / pow(2, 3 * levelmin_);
    }
  }

  template <typename T>
  void __write_dm_position(int coord, const grid_hierarchy &gh)
  {
    countLeafCells(gh);

    // update header
    hsize_t offset_fine = 0, offset_coarse = 0;

    for (unsigned i = 0; i < numFiles; i++)
    {
      hsize_t dims_fine = ceil(npfine / numFiles);
      hsize_t dims_coarse = ceil(npcoarse / numFiles);

      if (i == numFiles - 1)
      {
        dims_fine = npfine - offset_fine;
        dims_coarse = npcoarse - offset_coarse;
      }

      nPart[i][HIGHRES_DM_PARTTYPE] = dims_fine;
      nPart[i][coarsePartType] = dims_coarse;

      offset_fine += dims_fine;
      offset_coarse += dims_coarse;
    }

    nPartTotal[HIGHRES_DM_PARTTYPE] = npfine;
    nPartTotal[coarsePartType] = npcoarse;

    // FINE: collect displacements and convert to absolute coordinates with correct units
    int ilevel = gh.levelmax();

    std::vector<T> data(npfine);
    std::vector<T> lagrangian(npfine);
    size_t count = 0;

    for (unsigned i = 0; i < gh.get_grid(ilevel)->size(0); ++i)
      for (unsigned j = 0; j < gh.get_grid(ilevel)->size(1); ++j)
        for (unsigned k = 0; k < gh.get_grid(ilevel)->size(2); ++k)
          if (gh.is_in_mask(ilevel, i, j, k) && !gh.is_refined(ilevel, i, j, k))
          {
            double xx[3];
            gh.cell_pos(ilevel, i, j, k, xx);

            xx[coord] += coordinateShift[coord];
            maximumDisplacement[coord] = std::max(maximumDisplacement[coord],
                std::fabs((*gh.get_grid(ilevel))(i, j, k) * boxSize * posFac));
            lagrangian[count] = (T)(fmod(xx[coord] * boxSize + boxSize, boxSize) * posFac);
            xx[coord] = (xx[coord] + (*gh.get_grid(ilevel))(i, j, k)) * boxSize;
            xx[coord] = fmod(xx[coord] + boxSize, boxSize);

            data[count++] = (T)(xx[coord] * posFac);
          }

    writeHDF5_b("Coordinates", coord, HIGHRES_DM_PARTTYPE, data); // write fine DM
    writeHDF5_b("LagrangianCoordinates", coord, HIGHRES_DM_PARTTYPE, lagrangian);
    // The common dyadic material cell is distinct from the staggered gas q
    // and the combined coarse species COM. Retain its geometry directly;
    // compensated particle masses cannot reconstruct the original volume.
    writeHDF5_b("LagrangianCellCenters", coord, HIGHRES_DM_PARTTYPE, lagrangian);
    if (coord == 0)
      writeHDF5_a("LagrangianCellWidths", HIGHRES_DM_PARTTYPE,
          std::vector<double>(npfine, boxSize * posFac / std::pow(2.0, gh.levelmax())));

    if (count != npfine)
      throw std::runtime_error("Internal consistency error while writing fine DM pos");

    // COARSE: collect displacements and convert to absolute coordinates with correct units
    if (levelmax_ != levelmin_) // multimass
    {
      data = std::vector<T>(npcoarse, 0.0);
      lagrangian = std::vector<T>(npcoarse, 0.0);
      std::vector<double> cellWidths(npcoarse, 0.0);
      count = 0;

      for (int ilevel = gh.levelmax() - 1; ilevel >= (int)gh.levelmin(); --ilevel)
        for (unsigned i = 0; i < gh.get_grid(ilevel)->size(0); ++i)
          for (unsigned j = 0; j < gh.get_grid(ilevel)->size(1); ++j)
            for (unsigned k = 0; k < gh.get_grid(ilevel)->size(2); ++k)
              if (gh.is_in_mask(ilevel, i, j, k) && !gh.is_refined(ilevel, i, j, k))
              {
                double xx[3];
                gh.cell_pos(ilevel, i, j, k, xx);

                xx[coord] += coordinateShift[coord];
                maximumDisplacement[coord] = std::max(maximumDisplacement[coord],
                    std::fabs((*gh.get_grid(ilevel))(i, j, k) * boxSize * posFac));
                lagrangian[count] = (T)(fmod(xx[coord] * boxSize + boxSize, boxSize) * posFac);
                cellWidths[count] = boxSize * posFac / std::pow(2.0, ilevel);
                xx[coord] = (xx[coord] + (*gh.get_grid(ilevel))(i, j, k)) * boxSize;

                if (!doBaryons) // if so, we will handle the mod in write_gas_position
                  xx[coord] = fmod(xx[coord] + boxSize, boxSize) * posFac;

                data[count++] = (T)xx[coord];
              }

      if (count != npcoarse)
        throw std::runtime_error("Internal consistency error while writing coarse DM pos");

      writeHDF5_b("Coordinates", coord, coarsePartType, data); // write coarse DM
      writeHDF5_b("LagrangianCoordinates", coord, coarsePartType, lagrangian);
      writeHDF5_b("LagrangianCellCenters", coord, coarsePartType, lagrangian);
      if (coord == 0) writeHDF5_a("LagrangianCellWidths", coarsePartType, cellWidths);
    }
  }

  template <typename T>
  void __write_dm_velocity(int coord, const grid_hierarchy &gh)
  {
    countLeafCells(gh);

    // FINE: collect velocities and convert to correct units
    int ilevel = gh.levelmax();

    std::vector<T> data(npfine);
    size_t count = 0;

    for (unsigned i = 0; i < gh.get_grid(ilevel)->size(0); ++i)
      for (unsigned j = 0; j < gh.get_grid(ilevel)->size(1); ++j)
        for (unsigned k = 0; k < gh.get_grid(ilevel)->size(2); ++k)
          if (gh.is_in_mask(ilevel, i, j, k) && !gh.is_refined(ilevel, i, j, k))
          {
            data[count++] = (T)(*gh.get_grid(ilevel))(i, j, k) * velFac;
          }

    writeHDF5_b("Velocities", coord, HIGHRES_DM_PARTTYPE, data); // write fine DM

    if (count != npfine)
      throw std::runtime_error("Internal consistency error while writing fine DM pos");

    // COARSE: collect velocities and convert to correct units
    if (levelmax_ != levelmin_) // multimass
    {
      data = std::vector<T>(npcoarse, 0.0);
      count = 0;

      for (int ilevel = gh.levelmax() - 1; ilevel >= (int)gh.levelmin(); --ilevel)
        for (unsigned i = 0; i < gh.get_grid(ilevel)->size(0); ++i)
          for (unsigned j = 0; j < gh.get_grid(ilevel)->size(1); ++j)
            for (unsigned k = 0; k < gh.get_grid(ilevel)->size(2); ++k)
              if (gh.is_in_mask(ilevel, i, j, k) && !gh.is_refined(ilevel, i, j, k))
              {
                data[count++] = (T)(*gh.get_grid(ilevel))(i, j, k) * velFac;
              }

      if (count != npcoarse)
        throw std::runtime_error("Internal consistency error while writing coarse DM pos");

      writeHDF5_b("Velocities", coord, coarsePartType, data); // write coarse DM
    }
  }

  template <typename T>
  void __write_gas_velocity(int coord, const grid_hierarchy &gh)
  {
    countLeafCells(gh);

    std::vector<T> gas_data(npart); // read/write gas at all levels from the gh
    size_t count = 0;

    for (int ilevel = levelmax_; ilevel >= (int)levelmin_; --ilevel)
      for (unsigned i = 0; i < gh.get_grid(ilevel)->size(0); ++i)
        for (unsigned j = 0; j < gh.get_grid(ilevel)->size(1); ++j)
          for (unsigned k = 0; k < gh.get_grid(ilevel)->size(2); ++k)
            if (gh.is_in_mask(ilevel, i, j, k) && !gh.is_refined(ilevel, i, j, k))
            {
              gas_data[count++] = (T)(*gh.get_grid(ilevel))(i, j, k) * velFac;
            }

    if (count != npart)
      throw std::runtime_error("Internal consistency error while writing GAS pos");

    // calculate modified DM velocities if: multimass and baryons present
    if (doBaryons && npcoarse)
    {
      double facb = omega_b / omega0;
      double facc = (omega0 - omega_b) / omega0;

      std::vector<T> dm_data(npcoarse);

      writeHDF5_b("Velocities", coord, coarsePartType, dm_data, true); // read coarse DM vels

      // overwrite
      for (size_t i = 0; i < npcoarse; i++)
        dm_data[i] = facc * dm_data[i] + facb * gas_data[npfine + i];

      writeHDF5_b("Velocities", coord, coarsePartType, dm_data); // overwrite coarse DM vels
    }                                                            // dm_data deallocated

    // restrict gas_data to fine only and request write
    std::vector<T> data(gas_data.begin() + 0, gas_data.begin() + npfine);

    std::vector<T>().swap(gas_data); // deallocate

    writeHDF5_b("Velocities", coord, GAS_PARTTYPE, data); // write highres gas
  }

  template <typename T>
  void __write_gas_position(int coord, const grid_hierarchy &gh)
  {
    countLeafCells(gh);

    // update header (will actually write only gas at levelmax)
    hsize_t offset = 0;

    for (unsigned i = 0; i < numFiles; i++)
    {
      hsize_t dims = ceil(npfine / numFiles);
      if (i == numFiles - 1)
        dims = npfine - offset;

      nPart[i][GAS_PARTTYPE] = dims;
      offset += dims;
    }

    nPartTotal[GAS_PARTTYPE] = npfine;

    std::vector<double> gas_data(npart); // read/write gas at all levels from the gh
    std::vector<T> gas_lagrangian(npfine);
    size_t count = 0;

    double h = 1.0 / (1ul << gh.levelmax());

    for (int ilevel = gh.levelmax(); ilevel >= (int)gh.levelmin(); --ilevel)
      for (unsigned i = 0; i < gh.get_grid(ilevel)->size(0); ++i)
        for (unsigned j = 0; j < gh.get_grid(ilevel)->size(1); ++j)
          for (unsigned k = 0; k < gh.get_grid(ilevel)->size(2); ++k)
            if (gh.is_in_mask(ilevel, i, j, k) && !gh.is_refined(ilevel, i, j, k))
            {
              double xx[3];
              gh.cell_pos(ilevel, i, j, k, xx);
              xx[coord] += coordinateShift[coord];
              maximumDisplacement[coord] = std::max(maximumDisplacement[coord],
                  std::fabs((*gh.get_grid(ilevel))(i, j, k) * boxSize * posFac));

              // shift particle positions (this has to be done as the same shift
              // is used when computing the convolution kernel for SPH baryons)
              xx[coord] += 0.5 * h;
              if (count < npfine)
                gas_lagrangian[count] = (T)(fmod(xx[coord] * boxSize + boxSize, boxSize) * posFac);

              xx[coord] = (xx[coord] + (*gh.get_grid(ilevel))(i, j, k)) * boxSize;

              gas_data[count++] = xx[coord];
            }

    if (count != npart)
      throw std::runtime_error("Internal consistency error while writing coarse DM pos");

    // calculate modified DM coordinates if: multimass and baryons present
    if (doBaryons && npcoarse)
    {
      const double facb = omega_b / omega0;
      const double facc = (omega0 - omega_b) / omega0;
      std::vector<T> dm_data(npcoarse);

      writeHDF5_b("Coordinates", coord, coarsePartType, dm_data, true); // read coarse DM vels

      // overwrite
      for (size_t i = 0; i < npcoarse; i++)
      {
        dm_data[i] = facc * dm_data[i] + facb * gas_data[npfine + i];
        dm_data[i] = fmod(dm_data[i] + boxSize, boxSize) * posFac;
      }

      writeHDF5_b("Coordinates", coord, coarsePartType, dm_data); // overwrite coarse DM vels
      // Coarse particles carry the baryon+CDM center of mass. Their q uses
      // the same baryonic staggering as x, with no displacement contribution.
      // Reuse the output vector to reconstruct original coarse q through the
      // same unwrapped COM arithmetic as x. Adding a weighted stagger to an
      // already wrapped/scaled q follows a different rounding path at zero flow.
      count = 0;
      for (int ilevel = gh.levelmax() - 1; ilevel >= (int)gh.levelmin(); --ilevel)
        for (unsigned i = 0; i < gh.get_grid(ilevel)->size(0); ++i)
          for (unsigned j = 0; j < gh.get_grid(ilevel)->size(1); ++j)
            for (unsigned k = 0; k < gh.get_grid(ilevel)->size(2); ++k)
              if (gh.is_in_mask(ilevel, i, j, k) && !gh.is_refined(ilevel, i, j, k))
              {
                double xx[3];gh.cell_pos(ilevel, i, j, k, xx);
                xx[coord] += coordinateShift[coord];
                const T original_dm = (T)(xx[coord] * boxSize);
                xx[coord] += 0.5 * h;
                const T combined = (T)(facc * original_dm + facb * (xx[coord] * boxSize));
                dm_data[count++] = (T)(fmod(combined + boxSize, boxSize) * posFac);
              }
      if (count != npcoarse)
        throw std::runtime_error("Incomplete original coarse COM coordinates");
      writeHDF5_b("LagrangianCoordinates", coord, coarsePartType, dm_data);
    }

    // restrict gas_data to fine only and request write
    //std::vector<float> data( gas_data.begin() + 0, gas_data.begin() + npfine );

    std::vector<T> data(npfine);

    for (size_t i = 0; i < npfine; i++)
      data[i] = (T)(fmod(gas_data[i] + boxSize, boxSize) * posFac);

    std::vector<double>().swap(gas_data); // deallocate

    writeHDF5_b("Coordinates", coord, GAS_PARTTYPE, data); // write highres gas
    writeHDF5_b("LagrangianCoordinates", coord, GAS_PARTTYPE, gas_lagrangian);
    std::vector<T> commonCenters(npfine);
    writeHDF5_b("LagrangianCellCenters", coord, HIGHRES_DM_PARTTYPE, commonCenters, true);
    writeHDF5_b("LagrangianCellCenters", coord, GAS_PARTTYPE, commonCenters);
    if (coord == 0)
      writeHDF5_a("LagrangianCellWidths", GAS_PARTTYPE,
          std::vector<double>(npfine, boxSize * posFac / std::pow(2.0, gh.levelmax())));
  }

public:
  arepo_output_plugin(config_file &cf) : output_plugin(cf)
  {
    // ensure that everyone knows we want to do SPH, implies: bsph=1, bbshift=1, decic_baryons=1
    // -> instead of just writing gas densities (which are here ignored), the gas displacements are also written
    cf.insert_value("setup", "do_SPH", "yes");
    // The hierarchy internally recenters zooms by integer coarse cells.
    // Restore the requested parent frame for both x and q on output. The
    // noise plugin already evaluates the correspondingly translated phases.
    shiftBack = cf.get_value_safe<bool>("output", "shift_back", true);
    coordinateShift = std::vector<double>(3, 0.0);
    maximumDisplacement = std::vector<double>(3, 0.0);
    if (shiftBack)
    {
      const double coarse_spacing = 1.0 / std::pow(2.0, cf.get_value<int>("setup", "levelmin"));
      coordinateShift[0] = -cf.get_value_safe<int>("setup", "shift_x", 0) * coarse_spacing;
      coordinateShift[1] = -cf.get_value_safe<int>("setup", "shift_y", 0) * coarse_spacing;
      coordinateShift[2] = -cf.get_value_safe<int>("setup", "shift_z", 0) * coarse_spacing;
    }

    // init header and config parameters
    nPartTotal = std::vector<long long>(NTYPES, 0);
    massTable = std::vector<double>(NTYPES, 0.0);

    coarsePartType = cf.get_value_safe<unsigned>("output", "arepo_coarsetype", COARSE_DM_DEFAULT_PARTTYPE);
    UnitLength_in_cm = cf.get_value_safe<double>("output", "arepo_unitlength", 3.0856775814913673e21); // project kpc
    UnitMass_in_g = cf.get_value_safe<double>("output", "arepo_unitmass", 1.98847e43);         // project 1e10 Msun
    UnitVelocity_in_cm_per_s = cf.get_value_safe<double>("output", "arepo_unitvel", 1e5);    // 1 km/sec

    omega0 = cf.get_value<double>("cosmology", "Omega_m");
    omega_b = cf.get_value<double>("cosmology", "Omega_b");
    omega_L = cf.get_value<double>("cosmology", "Omega_L");
    redshift = cf.get_value<double>("setup", "zstart");
    boxSize = cf.get_value<double>("setup", "boxlength");
    doBaryons = cf.get_value_safe<bool>("setup", "baryons", false);
    useLongIDs = cf.get_value_safe<bool>("output", "arepo_longids", false);
    numFiles = cf.get_value_safe<unsigned>("output", "arepo_num_files", 1);
    doublePrec = cf.get_value_safe<bool>("output", "arepo_doubleprec", 0);
    if(!numFiles)throw std::runtime_error("Native IC output needs a positive file count");
    if(cf.get_value_safe<std::string>("cosmology","transfer","")=="DMS_file" &&
       (numFiles!=1 || !doublePrec || !useLongIDs))
      throw std::runtime_error("DMS native ICs require one double-precision file, uint64 IDs and full GADGET-4 counts");

    for (unsigned i = 0; i < numFiles; i++)
      nPart.push_back(std::vector<unsigned long long>(NTYPES, 0));

    // factors which multiply positions and velocities
    time = 1.0 / (1.0 + redshift);
    posFac = 3.0856775814913673e24 / UnitLength_in_cm; // project Mpc to native length
    velFac = (1.0f / sqrt(time)) * boxSize;

    // critical density
    // Same G and unit conventions as the project's GADGET backend. Derive
    // critical density rather than importing an older rounded mass constant.
    const double H100_cgs = 1.0e7 / 3.0856775814913673e24;
    rhoCrit = 3.0 * H100_cgs * H100_cgs / (8.0 * M_PI * 6.67430e-8);
    rhoCrit *= pow(UnitLength_in_cm, 3.0) / UnitMass_in_g;

    // calculate PMGRID suggestion
    pmgrid = pow(2, levelmin_) * 2; // unigrid
    gridboost = 1;

    if (levelmin_ != levelmax_)
    {
      vec3_t lxref, x0ref, x1ref;
      double pmgrid_new;

      the_region_generator->get_AABB(x0ref, x1ref, levelmax_); // generalized beyond box
      for (int i = 0; i < 3; i++)
        lxref[i] = x1ref[i] - x0ref[i];

      // fraction box length of the zoom region
      lxref[0] = pow((lxref[0] * lxref[1] * lxref[2]), 0.333);

      pmgrid_new = pow(2, levelmax_) * 2; // to cover entire box at highest resolution
      pmgrid_new *= lxref[0];             // only need to cover a fraction

      if ((gridboost = round(pmgrid_new / pmgrid)) > 1)
        gridboost = pow(2, ceil(log(gridboost) / log(2.0))); // round to nearest, higher power of 2
      if (gridboost == 0)
        gridboost = 1;
    }

    // calculate Tini for gas
    hubbleParam = cf.get_value<double>("cosmology", "H0") / 100.0;

    double astart = 1.0 / (1.0 + redshift);
    double h2 = hubbleParam * hubbleParam;
    double adec = 1.0 / (160.0 * pow(omega_b * h2 / 0.022, 2.0 / 5.0));
    double Tcmb0 = 2.726;

    Tini = astart < adec ? Tcmb0 / astart : Tcmb0 / astart / astart * adec;

    // calculate softening suggestion
    softening = (boxSize * posFac) / pow(2, levelmax_) / 40.0;

    // header and sanity checks
    if (!doBaryons)
      massTable[HIGHRES_DM_PARTTYPE] = omega0 * rhoCrit * pow(boxSize * posFac, 3.0) / pow(2, 3 * levelmax_);
    else
      massTable[HIGHRES_DM_PARTTYPE] = (omega0 - omega_b) * rhoCrit * pow(boxSize * posFac, 3.0) / pow(2, 3 * levelmax_);

    if (coarsePartType == GAS_PARTTYPE || coarsePartType == HIGHRES_DM_PARTTYPE)
      throw std::runtime_error("Error: Specified illegal Arepo particle type for coarse particles.");
    if (coarsePartType == STAR_PARTTYPE)
      music::wlog.Print("WARNING: Specified coarse particle type will collide with stars if USE_SFR enabled.");

    // create file(s)
    for (unsigned i = 0; i < numFiles; i++)
    {
      std::string filename = fname_;
      if (numFiles > 1)
      {
        size_t pos = filename.find(".hdf5");
        if (pos != filename.length() - 5)
          throw std::runtime_error("Error: Unexpected output filename (doesn't end in .hdf5).");

        std::stringstream s;
        s << "." << i << ".hdf5";
        filename.replace(pos, 5, s.str());
      }

      HDFCreateFile(filename);

      // create particle type groups
      std::stringstream GrpName;
      GrpName << "PartType" << HIGHRES_DM_PARTTYPE;

      HDFCreateGroup(filename, GrpName.str().c_str()); // highres or unigrid DM

      if (doBaryons)
      {
        GrpName.str("");
        GrpName << "PartType" << GAS_PARTTYPE;
        HDFCreateGroup(filename, GrpName.str().c_str()); // gas
      }

      if (levelmax_ != levelmin_) // multimass
      {
        GrpName.str("");
        GrpName << "PartType" << coarsePartType;
        HDFCreateGroup(filename, GrpName.str().c_str()); // coarse DM
      }
    }
  }

  ~arepo_output_plugin()
  {
  }

  // A nested quadrature need not integrate a zero-mode-free grid to zero
  // particle momentum. Remove only the global velocity zero mode from all
  // delivered species together, preserving relative velocities. The applied
  // frame shift is saved explicitly in the native header.
  void removeDeliveredMomentumZeroMode()
  {
    if (cf_.get_value_safe<std::string>("cosmology", "transfer", "") != "DMS_file")
      return;
    if (numFiles != 1 || !doublePrec)
      throw std::runtime_error("DMS particle momentum closure requires one double-precision native IC");
    const auto readMasses = [&](int type, size_t count) {
      std::vector<double> values(count, massTable[type]);
      if (massTable[type] == 0.0)
      {
        const std::string name = "PartType" + std::to_string(type) + "/Masses";
        hid_t file = H5Fopen(fname_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t dataset = file < 0 ? -1 : H5Dopen(file, name.c_str());
        herr_t status = dataset < 0 ? -1 : H5Dread(dataset, H5T_NATIVE_DOUBLE,
            H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
        if (dataset >= 0) H5Dclose(dataset);
        if (file >= 0) H5Fclose(file);
        if (status < 0) throw std::runtime_error("Cannot read native masses for particle momentum closure");
      }
      for (double value : values)
        if (!(std::isfinite(value) && value > 0.0))
          throw std::runtime_error("Invalid native mass in particle momentum closure");
      return values;
    };
    long double total = 0.0L;
    std::array<long double, 3> momentum = {{0.0L, 0.0L, 0.0L}};
    for (int type : {GAS_PARTTYPE, HIGHRES_DM_PARTTYPE, (int)coarsePartType})
    {
      const size_t count = nPartTotal[type];
      if (!count) continue;
      const auto masses = readMasses(type, count);
      for (double mass : masses) total += (long double)mass;
      std::vector<double> values(count);
      for (int coord = 0; coord < 3; ++coord)
      {
        writeHDF5_b("Velocities", coord, type, values, true);
        for (size_t index = 0; index < count; ++index)
        {
          if (!std::isfinite(values[index]))
            throw std::runtime_error("Invalid native velocity in particle momentum closure");
          momentum[coord] += (long double)masses[index] * values[index];
        }
      }
    }
    if (!(total > 0.0L && std::isfinite(total)))
      throw std::runtime_error("No finite native mass for particle momentum closure");
    for (int coord = 0; coord < 3; ++coord)
      removedBulkVelocity[coord] = (double)(momentum[coord] / total);
    for (int type : {GAS_PARTTYPE, HIGHRES_DM_PARTTYPE, (int)coarsePartType})
    {
      const size_t count = nPartTotal[type];
      if (!count) continue;
      std::vector<double> values(count);
      for (int coord = 0; coord < 3; ++coord)
      {
        writeHDF5_b("Velocities", coord, type, values, true);
        for (double &value : values) value -= removedBulkVelocity[coord];
        writeHDF5_b("Velocities", coord, type, values);
      }
    }
  }

  void writeCompensatedMass(const grid_hierarchy &gh, int type)
  {
    countLeafCells(gh);
    const int level = gh.levelmax();
    std::vector<double> contrast;
    contrast.reserve(npfine);
    long double sum = 0.0L;
    for (unsigned i = 0; i < gh.get_grid(level)->size(0); ++i)
      for (unsigned j = 0; j < gh.get_grid(level)->size(1); ++j)
        for (unsigned k = 0; k < gh.get_grid(level)->size(2); ++k)
          if (gh.is_in_mask(level,i,j,k) && !gh.is_refined(level,i,j,k))
          {
            const double value = (*gh.get_grid(level))(i,j,k);
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite compensated species density");
            contrast.push_back(value); sum += value;
          }
    if (contrast.size() != npfine || !npfine)
      throw std::runtime_error("Incomplete compensated species density");
    // Explicit finite-window zero mode: keep each represented species mass
    // equal to its cosmological mean. Never normalize a Fourier amplitude.
    const double mean = (double)(sum / npfine);
    const double fraction = type == GAS_PARTTYPE ? omega_b : omega0-omega_b;
    const double meanMass = fraction*rhoCrit*pow(boxSize*posFac,3.0)/pow(2,3*level);
    for (double &value : contrast)
    {
      value = meanMass*(1.0+value-mean);
      if (!(std::isfinite(value) && value > 0.0))
        throw std::runtime_error("Compensated species mode produces nonpositive particle mass");
    }
    compensationMean[type] = mean;
    meanSpeciesMass[type] = meanMass;
    massTable[type] = 0.0;
    writeHDF5_a("Masses",type,contrast);
  }

  void write_lpt_component(const std::string &name, int coord, const grid_hierarchy &gh,
                           bool gas, bool velocity) override
  {
    countLeafCells(gh);
    std::vector<double> values(npart);
    size_t count = 0;
    const double scale = velocity ? velFac : boxSize*posFac;
    for (int level=gh.levelmax(); level >= (int)gh.levelmin(); --level)
      for (unsigned i=0;i<gh.get_grid(level)->size(0);++i)
        for (unsigned j=0;j<gh.get_grid(level)->size(1);++j)
          for (unsigned k=0;k<gh.get_grid(level)->size(2);++k)
            if (gh.is_in_mask(level,i,j,k) && !gh.is_refined(level,i,j,k))
              values[count++] = (*gh.get_grid(level))(i,j,k)*scale;
    if (count != npart) throw std::runtime_error("Incomplete retained LPT component field");
    std::vector<double> fine(values.begin(),values.begin()+npfine);
    writeHDF5_b(name,coord,gas?GAS_PARTTYPE:HIGHRES_DM_PARTTYPE,fine);
    if (npcoarse)
    {
      std::vector<double> coarse(values.begin()+npfine,values.end());
      if (gas)
      {
        std::vector<double> previous(npcoarse);
        writeHDF5_b(name,coord,coarsePartType,previous,true);
        for (size_t index=0;index<npcoarse;++index)
          coarse[index] = (1-omega_b/omega0)*previous[index]+(omega_b/omega0)*coarse[index];
      }
      writeHDF5_b(name,coord,coarsePartType,coarse);
    }
  }

  /* ------------------------------------------------------------------------------- */
  void write_dm_mass(const grid_hierarchy &gh)
  {
    if (!doublePrec)
      __write_dm_mass<float>(gh);
    else
      __write_dm_mass<double>(gh);
  }

  void write_dm_position(int coord, const grid_hierarchy &gh)
  {
    if (!doublePrec)
      __write_dm_position<float>(coord, gh);
    else
      __write_dm_position<double>(coord, gh);
  }

  void write_dm_velocity(int coord, const grid_hierarchy &gh)
  {
    if (!doublePrec)
      __write_dm_velocity<float>(coord, gh);
    else
      __write_dm_velocity<double>(coord, gh);
  }

  void write_dm_density(const grid_hierarchy &gh)
  {
    if (cf_.get_value_safe<bool>("cosmology","mixed_2lpt",false))
      writeCompensatedMass(gh,HIGHRES_DM_PARTTYPE);
  }

  void write_dm_potential(const grid_hierarchy &gh)
  { /* skip */
  }

  /* ------------------------------------------------------------------------------- */
  void write_gas_velocity(int coord, const grid_hierarchy &gh)
  {
    if (!doublePrec)
      __write_gas_velocity<float>(coord, gh);
    else
      __write_gas_velocity<double>(coord, gh);
  }

  void write_gas_position(int coord, const grid_hierarchy &gh)
  {
    if (!doublePrec)
      __write_gas_position<float>(coord, gh);
    else
      __write_gas_position<double>(coord, gh);
  }

  void write_gas_density(const grid_hierarchy &gh)
  {
    if (cf_.get_value_safe<bool>("cosmology","mixed_2lpt",false))
    {
      writeCompensatedMass(gh,GAS_PARTTYPE);
      return;
    }
    // if only saving highres gas, then all gas cells have the same initial mass
    // do not write out densities as we write out displacements
    if (doBaryons)
      massTable[GAS_PARTTYPE] = omega_b * rhoCrit * pow(boxSize * posFac, 3.0) / pow(2, 3 * levelmax_);
  }

  void write_gas_potential(const grid_hierarchy &gh)
  { /* skip */
  }

  void finalize(void)
  {
    // generate and add contiguous IDs for each particle type we have written
    removeDeliveredMomentumZeroMode();
    generateAndWriteIDs();

    // output particle counts
    std::cout << " - Arepo : wrote " << nPartTotAllTypes << " particles..." << std::endl;
    for (size_t i = 0; i < nPartTotal.size(); i++)
      std::cout << "    type [" << i << "] : " << std::setw(12) << nPartTotal[i] << std::endl;
    std::cout << std::endl;

    // write final header (some of these fields are required, others are extra info)
    for (unsigned i = 0; i < numFiles; i++)
    {
      std::string filename = fname_;
      if (numFiles > 1)
      {
        std::stringstream s;
        s << "." << i << ".hdf5";
        filename.replace(filename.find(".hdf5"), 5, s.str());

        std::cout << "    " << filename;
        for (size_t j = 0; j < nPart[i].size(); j++)
          std::cout << " " << std::setw(10) << nPart[i][j];
        std::cout << std::endl;
      }

      HDFCreateGroup(filename, "Header");

      music_particle_counts::write(filename,nPart[i],nPartTotal,
          cf_.get_value_safe<std::string>("cosmology","transfer","")=="DMS_file");
      HDFWriteGroupAttribute(filename, "Header", "MassTable", massTable);
      HDFWriteGroupAttribute(filename, "Header", "BoxSize", boxSize * posFac);
      HDFWriteGroupAttribute(filename, "Header", "NumFilesPerSnapshot", numFiles);
      HDFWriteGroupAttribute(filename, "Header", "Time", time);
      HDFWriteGroupAttribute(filename, "Header", "Redshift", redshift);
      HDFWriteGroupAttribute(filename, "Header", "Omega0", omega0);
      HDFWriteGroupAttribute(filename, "Header", "OmegaLambda", omega_L);
      HDFWriteGroupAttribute(filename, "Header", "OmegaBaryon", omega_b);
      HDFWriteGroupAttribute(filename, "Header", "HubbleParam", hubbleParam);
      HDFWriteGroupAttribute(filename, "Header", "Flag_Sfr", 0);
      HDFWriteGroupAttribute(filename, "Header", "Flag_Cooling", 0);
      HDFWriteGroupAttribute(filename, "Header", "Flag_StellarAge", 0);
      HDFWriteGroupAttribute(filename, "Header", "Flag_Metals", 0);
      HDFWriteGroupAttribute(filename, "Header", "Flag_Feedback", 0);
      HDFWriteGroupAttribute(filename, "Header", "Flag_DoublePrecision", (int)doublePrec);
      HDFWriteGroupAttribute(filename, "Header", "Music_levelmin", levelmin_);
      // Schema 1: pre-displacement periodic q, in the same comoving units
      // and exact ParticleID row order as Coordinates; coarse q is the COM.
      HDFWriteGroupAttribute(filename, "Header", "Music_lagrangian_schema", 1);
      HDFWriteGroupAttribute(filename, "Header", "Music_material_cell_schema", 1);
      HDFWriteGroupAttribute(filename, "Header", "Music_material_cell_unitlength_cm", UnitLength_in_cm);
      HDFWriteGroupAttribute(filename, "Header", "Music_baryon_material_offset_native",
          std::vector<double>(3, doBaryons ? 0.5 * boxSize * posFac / std::pow(2.0, levelmax_) : 0.0));
      HDFWriteGroupAttribute(filename, "Header", "Music_lpt_order", cf_.get_value_safe<bool>("setup", "use_2LPT", false) ? 2 : 1);
      HDFWriteGroupAttribute(filename, "Header", "Music_lagrangian_unitlength_cm", UnitLength_in_cm);
      HDFWriteGroupAttribute(filename, "Header", "Music_shift_back", (int)shiftBack);
      HDFWriteGroupAttribute(filename, "Header", "Music_max_abs_displacement_native", maximumDisplacement);
      HDFWriteGroupAttribute(filename, "Header", "Music_output_coordinate_shift_box_fraction", coordinateShift);
      HDFWriteGroupAttribute(filename, "Header", "Music_removed_bulk_velocity_gadget", removedBulkVelocity);
      HDFWriteGroupAttribute(filename, "Header", "Music_momentum_schema", 1);
      if (cf_.get_value_safe<bool>("cosmology","mixed_2lpt",false))
      {
        HDFWriteGroupAttribute(filename,"Header","Music_mixed_fluid_schema",1);
        HDFWriteGroupAttribute(filename,"Header","Music_lpt_component_schema",1);
        HDFWriteGroupAttribute(filename,"Header","Music_composite_source_normalization_schema",1);
        HDFWriteGroupAttribute(filename,"Header","Music_compensation_window_zero_modes",compensationMean);
        HDFWriteGroupAttribute(filename,"Header","Music_mean_species_masses_native",meanSpeciesMass);
        HDFWriteGroupAttribute(filename,"Header","Music_2lpt_displacement_coefficient",
            cf_.get_value<double>("cosmology","lpt2_displacement"));
        HDFWriteGroupAttribute(filename,"Header","Music_2lpt_velocity_coefficient",
            cf_.get_value<double>("cosmology","lpt2_velocity"));
      }
      HDFWriteGroupAttribute(filename, "Header", "Music_levelmax", levelmax_);
      HDFWriteGroupAttribute(filename, "Header", "Music_levelcounts", levelcounts);
      HDFWriteGroupAttribute(filename, "Header", "haveBaryons", (int)doBaryons);
      HDFWriteGroupAttribute(filename, "Header", "longIDs", (int)useLongIDs);
      HDFWriteGroupAttribute(filename, "Header", "suggested_pmgrid", pmgrid);
      HDFWriteGroupAttribute(filename, "Header", "suggested_gridboost", gridboost);
      HDFWriteGroupAttribute(filename, "Header", "suggested_highressoft", softening);
      HDFWriteGroupAttribute(filename, "Header", "suggested_gas_Tinit", Tini);
    }

    // give config/parameter file hints
    if (useLongIDs)
      std::cout << " - Arepo: Wrote 64bit IDs, enable LONGIDS." << std::endl;
    if (doublePrec)
      std::cout << " - Arepo: Double precision ICs, set INPUT_IN_DOUBLEPRECISION." << std::endl;
    if (NTYPES != 6)
      std::cout << " - Arepo: Using [" << NTYPES << "] particle types, set NTYPES to match." << std::endl;
    if (doBaryons)
      std::cout << " - Arepo: Wrote high-res gas (only), set REFINEMENT_HIGH_RES_GAS and GENERATE_GAS_IN_ICS with "
                << "SPLIT_PARTICLE_TYPE=" << pow(2, coarsePartType) << "." << std::endl;
    if (levelmax_ != levelmin_)
      std::cout << " - Arepo: Have zoom type ICs, set PLACEHIGHRESREGION=" << pow(2, HIGHRES_DM_PARTTYPE)
                << " (suggest PMGRID=" << pmgrid << " with GRIDBOOST=" << gridboost << ")." << std::endl;
    else
      std::cout << " - Arepo: Have unigrid type ICs (suggest PMGRID=" << pmgrid << ")." << std::endl;
    if (levelmax_ > levelmin_ + 1)
      std::cout << " - Arepo: More than one coarse DM mass using same type, set INDIVIDUAL_GRAVITY_SOFTENING="
                << pow(2, coarsePartType) << " (+" << pow(2, STAR_PARTTYPE) << " if including stars)." << std::endl;
    if (doBaryons)
      std::cout << " - Arepo: Set initial gas temperature to " << std::fixed << std::setprecision(3) << Tini << " K." << std::endl;
    std::cout << " - Arepo: Suggest grav softening = " << std::setprecision(3) << softening << " for high res DM." << std::endl;
  }
};

namespace
{
output_plugin_creator_concrete<arepo_output_plugin> creator("arepo");
}

#endif // HAVE_HDF5
