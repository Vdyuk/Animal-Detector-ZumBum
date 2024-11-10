import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { ButtonModule } from 'primeng/button';
import { HomeComponent } from './home.component';
import { HomeRoutingModule } from './home-routing.module';
import { HttpClientModule } from '@angular/common/http';
import { MatTableModule } from '@angular/material/table';
import { MatButtonModule } from '@angular/material/button';
import { CarouselModule } from 'primeng/carousel';
import { TagModule } from 'primeng/tag';
import { HomeService } from './services/home.service';
import { FormsModule } from '@angular/forms';
import { MatDividerModule } from '@angular/material/divider';
import { MatCardModule } from '@angular/material/card';
import { MatInputModule } from '@angular/material/input';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatRadioModule } from '@angular/material/radio';
import { MatPaginatorModule } from '@angular/material/paginator';
import { MatFormFieldModule } from '@angular/material/form-field';

@NgModule({
  declarations: [
    HomeComponent
  ],
  imports: [
    MatTableModule,
    HttpClientModule,
    MatPaginatorModule,
    FormsModule,
    CommonModule,
    RouterModule,
    HomeRoutingModule,
    MatRadioModule,
    MatCheckboxModule,
    MatButtonModule,
    CarouselModule,
    MatDividerModule,
    TagModule,
    MatCardModule,
    ButtonModule,
    MatInputModule,
    MatFormFieldModule,
  ],
  providers: [HomeService]
})
export class HomeModule { }