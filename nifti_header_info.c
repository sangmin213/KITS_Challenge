struct nifti_1_header { /* NIFTI-1 usage         */  /* ANALYZE 7.5 field(s) */
                        /*************************/  /************************/

                                            /*--- was header_key substruct ---*/
    int   sizeof_hdr;    /*!< MUST be 348           */  /* int sizeof_hdr;      */
    char  data_type[10]; /*!< ++UNUSED++            */  /* char data_type[10];  */
    char  db_name[18];   /*!< ++UNUSED++            */  /* char db_name[18];    */
    int   extents;       /*!< ++UNUSED++            */  /* int extents;         */
    short session_error; /*!< ++UNUSED++            */  /* short session_error; */
    char  regular;       /*!< ++UNUSED++            */  /* char regular;        */
    char  dim_info;      /*!< MRI slice ordering.   */  /* char hkey_un0;       */

                                        /*--- was image_dimension substruct ---*/
    short dim[8];        /*!< Data array dimensions.*/  /* short dim[8];        */
    float intent_p1 ;    /*!< 1st intent parameter. */  /* short unused8;       */
                                                        /* short unused9;       */
    float intent_p2 ;    /*!< 2nd intent parameter. */  /* short unused10;      */
                                                        /* short unused11;      */
    float intent_p3 ;    /*!< 3rd intent parameter. */  /* short unused12;      */
                                                        /* short unused13;      */
    short intent_code ;  /*!< NIFTI_INTENT_* code.  */  /* short unused14;      */
    short datatype;      /*!< Defines data type!    */  /* short datatype;      */
    short bitpix;        /*!< Number bits/voxel.    */  /* short bitpix;        */
    short slice_start;   /*!< First slice index.    */  /* short dim_un0;       */
    float pixdim[8];     /*!< Grid spacings.        */  /* float pixdim[8];     */
    float vox_offset;    /*!< Offset into .nii file */  /* float vox_offset;    */
    float scl_slope ;    /*!< Data scaling: slope.  */  /* float funused1;      */
    float scl_inter ;    /*!< Data scaling: offset. */  /* float funused2;      */
    short slice_end;     /*!< Last slice index.     */  /* float funused3;      */
    char  slice_code ;   /*!< Slice timing order.   */
    char  xyzt_units ;   /*!< Units of pixdim[1..4] */
    float cal_max;       /*!< Max display intensity */  /* float cal_max;       */
    float cal_min;       /*!< Min display intensity */  /* float cal_min;       */
    float slice_duration;/*!< Time for 1 slice.     */  /* float compressed;    */
    float toffset;       /*!< Time axis shift.      */  /* float verified;      */
    int   glmax;         /*!< ++UNUSED++            */  /* int glmax;           */
    int   glmin;         /*!< ++UNUSED++            */  /* int glmin;           */

                                            /*--- was data_history substruct ---*/
    char  descrip[80];   /*!< any text you like.    */  /* char descrip[80];    */
    char  aux_file[24];  /*!< auxiliary filename.   */  /* char aux_file[24];   */

    short qform_code ;   /*!< NIFTI_XFORM_* code.   */  /*-- all ANALYZE 7.5 ---*/
    short sform_code ;   /*!< NIFTI_XFORM_* code.   */  /*   fields below here  */
                                                        /*   are replaced       */
    float quatern_b ;    /*!< Quaternion b param.   */
    float quatern_c ;    /*!< Quaternion c param.   */
    float quatern_d ;    /*!< Quaternion d param.   */
    float qoffset_x ;    /*!< Quaternion x shift.   */
    float qoffset_y ;    /*!< Quaternion y shift.   */
    float qoffset_z ;    /*!< Quaternion z shift.   */

    float srow_x[4] ;    /*!< 1st row affine transform.   */
    float srow_y[4] ;    /*!< 2nd row affine transform.   */
    float srow_z[4] ;    /*!< 3rd row affine transform.   */

    char intent_name[16];/*!< 'name' or meaning of data.  */

    char magic[4] ;      /*!< MUST be "ni1\0" or "n+1\0". */

} ;                   /**** 348 bytes total ****/