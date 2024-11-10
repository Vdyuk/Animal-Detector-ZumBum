interface ImageData {
    path: string;  // путь к изображению
    boundingBox: number[]; // массив с координатами ограничивающего прямоугольника
    success: number; // код успеха (1 - успех, 0 - неудача)
    modifiedPath: string | null; // измененный путь или null, если не изменялся
  }
  
  interface ResponseData {
    count_of_success: number; // количество успешных обработок
    count_of_failed: number; // количество неудачных обработок
    percent_of_success: number; // процент успешных обработок
    images: ImageData[]; // массив объектов ImageData
  }
  