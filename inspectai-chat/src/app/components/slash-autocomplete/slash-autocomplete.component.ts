import {
  Component, Input, Output, EventEmitter, OnChanges, SimpleChanges,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { SlashCommandService } from '../../services/slash-commands.service';

interface Suggestion {
  display:     string;
  completion:  string;
  description: string;
}

@Component({
  selector: 'app-slash-autocomplete',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="autocomplete-panel" *ngIf="suggestions.length > 0" role="listbox">
      <div
        *ngFor="let s of suggestions; let i = index"
        class="suggestion-row"
        [class.active]="i === activeIndex"
        (mousedown)="select(s)"
        (mouseover)="activeIndex = i"
        role="option"
        [attr.aria-selected]="i === activeIndex">
        <span class="cmd-text">{{ s.display }}</span>
        <span class="cmd-desc">{{ s.description }}</span>
      </div>
    </div>
  `,
  styles: [`
    .autocomplete-panel {
      position: absolute;
      bottom: calc(100% + 6px);
      left: 0; right: 0;
      background: #ffffff;
      border: 1px solid #e4e7ed;
      border-radius: 10px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.10), 0 2px 8px rgba(0,0,0,0.06);
      overflow: hidden;
      z-index: 100;
      max-height: 280px;
      overflow-y: auto;
    }
    .suggestion-row {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      padding: 9px 14px;
      cursor: pointer;
      border-bottom: 1px solid #f3f4f6;
      transition: background 0.1s;
    }
    .suggestion-row:last-child { border-bottom: none; }
    .suggestion-row.active { background: #edf9f8; }
    .cmd-text {
      font-family: ui-monospace, monospace;
      font-size: 13px;
      font-weight: 600;
      color: #3fbdad;
      white-space: nowrap;
    }
    .cmd-desc {
      font-size: 12px;
      color: #9aa0ad;
      font-family: ui-sans-serif, system-ui, sans-serif;
      text-align: right;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
  `],
})
export class SlashAutocompleteComponent implements OnChanges {

  @Input()  input  = '';
  @Output() chosen = new EventEmitter<string>();

  suggestions: Suggestion[] = [];
  activeIndex  = 0;

  constructor(private svc: SlashCommandService) {}

  ngOnChanges(c: SimpleChanges) {
    if (c['input']) {
      this.suggestions = this.svc.getSuggestions(this.input);
      this.activeIndex = 0;
    }
  }

  select(s: Suggestion) {
    this.chosen.emit(s.completion);
    this.suggestions = [];
  }

  handleKey(event: KeyboardEvent): boolean {
    if (!this.suggestions.length) return false;
    switch (event.key) {
      case 'ArrowUp':
        event.preventDefault();
        this.activeIndex = (this.activeIndex - 1 + this.suggestions.length) % this.suggestions.length;
        return true;
      case 'ArrowDown':
        event.preventDefault();
        this.activeIndex = (this.activeIndex + 1) % this.suggestions.length;
        return true;
      case 'Tab':
      case 'Enter':
        if (this.suggestions[this.activeIndex]) {
          event.preventDefault();
          this.select(this.suggestions[this.activeIndex]);
          return true;
        }
        return false;
      case 'Escape':
        this.suggestions = [];
        return true;
      default:
        return false;
    }
  }

  get isOpen(): boolean { return this.suggestions.length > 0; }
}
