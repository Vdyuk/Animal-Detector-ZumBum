import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { URLs } from '../../../base/urls';
import { Observable } from 'rxjs';

interface ImageData {
    path: string;
    boundingBox: number[];
    success: number;
    modifiedPath: string | null;
}

interface ResponseData {
    count_of_success: number;
    count_of_failed: number;
    percent_of_success: number;
    images: ImageData[];
}

@Injectable({
    providedIn: 'root'
})

export class HomeService {
    constructor(private _httpClient: HttpClient) { }

    uploadFiles(userId: string, min_width: number, min_height: number, upscale: boolean, files: File[]): Observable<any> {
        const formData: FormData = new FormData();

        // Добавляем файлы в FormData
        files.forEach(file => {
            formData.append('images', file, file.name); // 'images' как указано в Swagger
            formData.append('paths', file.webkitRelativePath);
        });

        return this._httpClient.post<ResponseData>(`${URLs.prediction.uploadandprocess}/?user_id=${userId}&min_width=${min_width}&min_height=${min_height}&upscale=${upscale}`, formData);
    }

    downloadPhoto(userId: string): Observable<Blob> {
        return this._httpClient.get(`${URLs.prediction.downloadprocessedimages}/?user_id=${userId}`, {
            responseType: 'blob' // Указываем, что ожидаем ответ в виде Blob
          });
    }

    deleteAcc(userId: string) {
        const url = `${URLs.prediction.deleteuserdata}/?user_id=${userId}`;
        const payload = new Blob([JSON.stringify({ user_id: userId })], {
            type: 'application/json',
          });
        if (navigator.sendBeacon) {
        
          navigator.sendBeacon(url, payload);
          
        } else {
          this._httpClient.post(url, payload);
        }
      }
    

    generateReport(userId: string): Observable<Blob> {
        return this._httpClient.get(`${URLs.prediction.generatereport}/?user_id=${userId}`, {
            responseType: 'blob' // Указываем, что ожидаем ответ в виде Blob
          });
    }
}

