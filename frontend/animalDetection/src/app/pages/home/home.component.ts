// home.component.ts
import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { HomeService } from './services/home.service';
import { MatSnackBar } from '@angular/material/snack-bar';
import { Destroyer } from '../../base/destroyer';
import { Router } from '@angular/router';
import { PhotoLoadService } from '../../shared/services/photo-load.service';


interface ImageData {
  path: string;  // путь к изображению
  success: number; // код успеха (1 - успех, 0 - неудача)
  relativePath: string | null;
  modifiedPath: string | null; // измененный путь или null, если не изменялся
}

interface ResponseData {

all_count: number;
  count_of_images_with_animals: number;
  count_of_animals_on_images: number;
  count_of_success: number;
  images: ImageData[]; // массив объектов ImageData
}



@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.less'],
})
export class HomeComponent extends Destroyer implements OnInit {

  messages: string[] = [
    'Цифровой прорыв 2024',
    'Автоматическая фильтрация изображений животных'
  ];
  currentMessageIndex: number = 0;
  intervalId: any;
  isLoading = false;

  file: File | null = null;
  fileList: FileList | null = null;
  isFileExist = false;
  allFiles: File[] = []; // Массив для хранения всех загруженных файлов
  user_key = '';
  selectedFile: File | null = null;
  photosList: ResponseData | null = null;
  @Output() photosLoaded = new EventEmitter<boolean>();
  upscale = false;
  showModal = false; // Переменная для управления видимостью модального окна
  currentImageUrl: string | null = null; // Переменная для текущего изображения

  totalPhotos: number | null = null;
  animalsCount: number | null = null;
  goodPhotosCount: number | null = null;


  displayedColumns: string[] = ['path', 'relativePath', 'modifiedPath'];
  photos: ImageData[] | null = null;

  min_width = 128;
  min_height = 128;

  constructor(
    private homeService: HomeService,
    private router: Router,
    private snackBar: MatSnackBar,
    private photoLoadService: PhotoLoadService
  ) {
    super();
    this.user_key = this.setUserKey()
  }

  ngOnInit() {
    this.intervalId = setInterval(() => {
      if (this.currentMessageIndex < this.messages.length) {
        document.getElementById('text-change')!.innerText = this.messages[this.currentMessageIndex];
        this.currentMessageIndex++;
      } else {
        // Остановить интервал после завершения всех сообщений
        clearInterval(this.intervalId);
        document.getElementById('text-change')!.innerText = ''; // Можно очистить текст, если нужно
      }
    }, 4000);
  }

  generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  setUserKey(): string {
    let userKey = localStorage.getItem('userKey');
    if (!userKey) {
      // Генерация уникального ключа 
      const uniqueKey = this.generateUUID();
      localStorage.setItem('userKey', uniqueKey);
      userKey = uniqueKey; // Assign uniqueKey to userKey if it was generated
    }
    return userKey; // Ensure a string is always returned.
  }

  validateDimensions() {
    if (this.min_width < 0) {
      this.min_width = 0;
    }
    if (this.min_height < 0) {
      this.min_height = 0;
    }
  }

  navigateToAnalysis() {
    if (this.allFiles.length) {
      this.isLoading = true;
      this.homeService.uploadFiles(this.user_key, this.min_width, this.min_height, this.upscale, this.allFiles)
        .subscribe({
          next: (response) => {
            if (response) {
              this.isLoading = false;
              this.photosList = response;
              this.photos = this.formatImages(response.images);
              this.photoLoadService.setPhotosLoaded(this.photos.length > 0);
            }
          },
          error: (error) => {
            // Обработка ошибок
            this.isLoading = false;
            this.openSnackBar('Ошибка при загрузке данных', 'Закрыть');
            this.photoLoadService.setPhotosLoaded(false);
          }
        });
    }
  }

  private formatImages(data: (string | (number[] | string | number | null))[][]): ImageData[] {
    return data.map(item => ({
      path: item[0] as string,
      success: item[1] as number,
      modifiedPath: item[2] as string | null,
      relativePath: item[3] as string | null,
    }));
  }


  openSnackBar(message: string, action: string) {
    this.snackBar.open(message, action, {
      duration: 5000,
      horizontalPosition: 'end',
      verticalPosition: 'top',
    });
  }

  onFilechange(event: any) {
    const files: FileList = event.target.files; // Получаем список выбранных файлов
    if (files.length) {
      Array.from(files).forEach(file => this.allFiles.push(file)); // Добавить файлы в общий массив
      this.updateFileCount(); // Обновляем счетчик
    }
  }

  onFolderChange(event: any) {
    const files: FileList = event.target.files; // Получаем список файлов из папки
    if (files.length) {
      Array.from(files).forEach(file => this.allFiles.push(file)); // Добавить файлы в общий массив
      this.updateFileCount(); // Обновляем счетчик
    }
  }



  openImage(imageUrl: string) {
    this.currentImageUrl = imageUrl; // Установите текущее изображение
    this.showModal = true; // Откройте модальное окно
  }

  closeModal() {
    this.showModal = false; // Закройте модальное окно
    this.currentImageUrl = null; // Сбросьте текущее изображение
  }

  onDragOver(event: DragEvent) {
    event.preventDefault(); // Предотвратить открытие файла в браузере
    event.stopPropagation();
    // Здесь можно изменить стиль метки или контейнера, чтобы показать, что файл можно перетащить
  }

  onDragLeave(event: DragEvent) {
    event.stopPropagation();
    // Восстановите стиль, когда файлы уходят
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();

    const items = event.dataTransfer?.items; // Получаем элементы, перетаскиваемые в дроп-зону 
    const promises: Promise<void>[] = []; // Массив для хранения промисов 

    if (items) {
      for (let i = 0; i < items.length; i++) {
        // Только если элемент является файлом 
        if (items[i].kind === 'file') {
          const fileEntry = items[i].webkitGetAsEntry(); // Получаем элемент как файловый элемент 
          if (fileEntry) {
            if (fileEntry.isFile) {
              const filePromise = new Promise<void>((resolve) => {
                const file = items[i].getAsFile();
                if (file) {
                  this.allFiles.push(file);
                  resolve(); // Успешно добавили файл 
                }
              });
              promises.push(filePromise); // Добавляем промис в массив
            } else if (fileEntry.isDirectory) {
              // Если это директория, рекурсивно обрабатываем её содержимое 
              this.handleDirectory(fileEntry);
            }
          }
        }
      }
    }

    Promise.all(promises).then(() => {
      this.updateFileCount(); // Обновляем счетчик после всех загрузок
    });
  }

  handleDirectory(directoryEntry: any) {
    const reader = directoryEntry.createReader();
    reader.readEntries((entries: any[]) => {
      const promises: Promise<void>[] = []; // Массив для хранения промисов

      entries.forEach((entry) => {
        if (entry.isFile) {
          // Если это файл, добавляем его
          const filePromise = new Promise<void>((resolve) => {
            entry.file((file: File) => {
              this.allFiles.push(file);
              resolve(); // Успешно добавили файл
            });
          });
          promises.push(filePromise); // Добавляем промис в массив
        } else if (entry.isDirectory) {
          // Если это подкаталог, рекурсивно вызываем этот метод
          this.handleDirectory(entry); // Здесь не добавляем промис, поскольку это изменение
        }
      });

      // Ждем завершения всех промисов и затем обновляем счетчик
      Promise.all(promises).then(() => {
        this.updateFileCount(); // Обновляем счетчик после всех загрузок
      });
    });
  }


  updateFileCount() {
    this.isFileExist = this.allFiles.length > 0; // Проверяем, есть ли файлы
  }


}
