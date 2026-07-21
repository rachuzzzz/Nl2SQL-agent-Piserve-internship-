import { Pipe, PipeTransform } from '@angular/core';

@Pipe({ name: 'colLabel', standalone: true })
export class ColLabelPipe implements PipeTransform {
  transform(col: string): string {
    return col.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }
}
