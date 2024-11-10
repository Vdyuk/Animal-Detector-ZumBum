import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ThemeService {
  private _theme: string;

  constructor() {
    this._theme = localStorage.getItem('theme') || 'default'; 
    document.documentElement.setAttribute('theme', this._theme); 
  }

  get theme(): string | null {
    return document.documentElement.getAttribute('theme');
  }

  set theme(name: string) {
    document.documentElement.setAttribute('theme', name);
  }
}