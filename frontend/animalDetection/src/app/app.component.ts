import { Component, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, RouterOutlet } from '@angular/router';
import { ThemeService } from '../shared/services/theme.service';
import { MatListModule } from '@angular/material/list';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { HomeModule } from './pages/home/home.module';
import { MatButtonModule } from '@angular/material/button';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { ButtonModule } from 'primeng/button';
import { HomeService } from './pages/home/services/home.service';
import { HttpClientModule } from '@angular/common/http';
import { MatDialog } from '@angular/material/dialog';
import { PhotoLoadService } from './shared/services/photo-load.service';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';


@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, HttpClientModule, MatProgressSpinnerModule, ButtonModule, FormsModule, MatSnackBarModule, ReactiveFormsModule, HomeModule, MatButtonModule,
    RouterOutlet, RouterModule, MatListModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.less']
})
export class AppComponent implements OnInit, OnDestroy {
  enabled = false;
  searchTerm = '';
  tooltipVisible = [false, false, false];
  tooltipText = ['', '', ''];
  isPhotosLoaded = false;
  clicksCount = 0;
  user_key = '';
  loadingReport = false; // Для отслеживания статуса загрузки отчета
  loadingPhotos = false;


  constructor(public dialog: MatDialog, private snackBar: MatSnackBar, private homeService: HomeService, private photoLoadService: PhotoLoadService, private _productService: HomeService, private _themeService: ThemeService) {
    this.setUserKey();

    const savedTheme = localStorage.getItem('theme');
    this.enabled = savedTheme === 'dark' || (savedTheme === null && this._themeService.theme === 'dark');

    document.body.classList.toggle('dark-mode', this.enabled);
    document.body.classList.toggle('light-mode', !this.enabled);

    this.photoLoadService.photosLoaded$.subscribe(loaded => {
      this.isPhotosLoaded = loaded;
    });


  }

  ngOnInit() {
    // Setting up beforeunload event listener
    window.addEventListener('beforeunload', this.handleBeforeUnload.bind(this));
  }

  ngOnDestroy() {
    // Cleanup the event listener
    window.removeEventListener('beforeunload', this.handleBeforeUnload.bind(this));
  }



  showTooltip(text: string, index: number) {
    this.tooltipText[index] = text;
    this.tooltipVisible[index] = true;
  }


  hideTooltip() {
    this.tooltipVisible.fill(false);
  }


  changeTheme() {
    this.enabled = !this.enabled;
    this._themeService.theme = this.enabled ? 'dark' : 'default';
    localStorage.setItem('theme', this.enabled ? 'dark' : 'default');
    document.body.classList.toggle('dark-mode', this.enabled);
    document.body.classList.toggle('light-mode', !this.enabled);
  }

  exportReport() {
    this.loadingReport = true;
    this.homeService.generateReport(this.user_key)
      .subscribe({
        next: (response) => {
          if (response) {
            // Создаем URL для загрузки файла
            const blob = new Blob([response], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
            const url = window.URL.createObjectURL(blob);

            // Создаем временную ссылку
            const a = document.createElement('a');
            a.href = url;
            a.download = 'history.xlsx'; // Имя файла по умолчанию
            document.body.appendChild(a);
            a.click(); // Инициируем клик по ссылке для скачивания
            a.remove(); // Удаляем временный элемент

            // Освобождаем созданный URL
            window.URL.revokeObjectURL(url);
          }

        },
        error: (error) => {

          // Обработка ошибки
          this.openSnackBar('Ошибка при загрузке данных', 'Закрыть'); // Показать уведомление об ошибке

        },
        complete: () => {
          this.loadingReport = false; // Окончание загрузки
        }

      });
  }

  exportPhotos() {
    this.loadingPhotos = true;
    this.homeService.downloadPhoto(this.user_key)
      .subscribe({
        next: (response) => {
          if (response) {
            // Создаем URL для загрузки файла
            const blob = new Blob([response], { type: 'application/zip' });
            const url = window.URL.createObjectURL(blob);

            // Создаем временную ссылку
            const a = document.createElement('a');
            a.href = url;
            a.download = 'processed_files.zip'; // Имя файла по умолчанию
            document.body.appendChild(a);
            a.click(); // Инициируем клик по ссылке для скачивания
            a.remove(); // Удаляем временный элемент

            // Освобождаем созданный URL
            window.URL.revokeObjectURL(url);

          }

        },
        error: (error) => {

          // Обработка ошибки
          this.openSnackBar('Ошибка при загрузке данных', 'Закрыть'); // Показать уведомление об ошибке

        },
        complete: () => {
          this.loadingPhotos = false; // Окончание загрузки
        }

      });
  }



  openSnackBar(message: string, action: string) {
    this.snackBar.open(message, action, {
      duration: 5000,
      horizontalPosition: 'end',
      verticalPosition: 'top',
    });
  }


  setUserKey() {
    const userKey = localStorage.getItem('userKey');
    if (!userKey) {
      const uniqueKey = this.generateUUID();
      localStorage.setItem('userKey', uniqueKey);
      this.user_key = uniqueKey;
    } else {
      this.user_key = userKey;
    }
  }

  handleBeforeUnload(event: BeforeUnloadEvent) {
    this.homeService.deleteAcc(this.user_key);  // No need to subscribe, just call
  }




  generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }



}