import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class PhotoLoadService {
  private photosLoadedSource = new BehaviorSubject<boolean>(false);
  // Observable для подписки
  photosLoaded$ = this.photosLoadedSource.asObservable();

  // Метод для обновления состояния загрузки фотографий
  setPhotosLoaded(loaded: boolean) {
    this.photosLoadedSource.next(loaded);
  }
}