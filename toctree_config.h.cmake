/*
 * Tucker-Octree multiresolution voxel data compression library
 * Copyright (C) 2024 CSC - IT Center for Science Ltd
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see
 * <https://www.gnu.org/licenses/>.
 */

/* 
 * Author: Juhani Kataja 
 * Affiliation: CSC - IT Center for Science Ltd
 * Email: juhani.kataja@csc.fi
 */


#ifndef CONFIG_H
#define CONFIG_H

#cmakedefine VDF_REAL_DTYPE @VDF_REAL_DTYPE@
#cmakedefine OCTREE_TUCKER_CORE_RANK @OCTREE_TUCKER_CORE_RANK@
#cmakedefine MAX_ROOT_DIMS
#cmakedefine OCTREE_VIEW_INDEX_TYPE
#cmakedefine OCTREE_RANGE_CHECK
#cmakedefine TOCTREE_ZFP_STREAM_ACCURACY @TOCTREE_ZFP_STREAM_ACCURACY@

#endif // CONFIG_H
